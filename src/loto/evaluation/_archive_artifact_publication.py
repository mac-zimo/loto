"""Crash-recoverable publication for the Task 2.3a artifact directory.

The parent-directory flock protects cooperative actors at the commit point.
Two complete hash passes over retained file descriptors, followed by metadata
checks and a fresh name-to-descriptor/hash check, then a terminal check of all
descriptors and names, detect observable concurrent mutation or replacement.
There remains a TOCTOU window after the last check against a non-cooperative
actor; no permanent immutability is guaranteed. Published directories are
never removed automatically.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile


AT_FDCWD = -100
RENAME_NOREPLACE = 1


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """Immutable identity and contents of one published artifact directory."""

    directory: Path
    st_dev: int
    st_ino: int
    file_hashes: tuple[tuple[str, str], ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(self.directory / filename for filename, _ in self.file_hashes)


class PublishedArtifactError(RuntimeError):
    """Failure after publication whose immutable receipt enables safe recovery."""

    __slots__ = ("_receipt",)

    def __init__(self, receipt: PublicationReceipt, cause: BaseException) -> None:
        super().__init__(
            "artifact directory was published but finalization failed after "
            f"{type(cause).__name__}: {cause}; recovery "
            f"requires inspection with receipt {receipt!r}"
        )
        object.__setattr__(self, "_receipt", receipt)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_receipt" and hasattr(self, "_receipt"):
            raise AttributeError("publication receipt is immutable")
        super().__setattr__(name, value)

    @property
    def receipt(self) -> PublicationReceipt:
        return self._receipt


@contextmanager
def lock_parent_directory(directory: Path):
    """Cooperatively lock the parent without creating a repository lock file."""

    parent = directory.parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename on Linux while refusing every existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise RuntimeError(
            "renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe publication"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(
            error_number,
            "official artifact destination already exists",
            destination,
        )
    if error_number == errno.ENOSYS:
        raise RuntimeError(
            "renameat2(RENAME_NOREPLACE) is unsupported; refusing unsafe publication"
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def staging_directories(directory: Path) -> tuple[Path, ...]:
    """Return interrupted sibling stagings associated with one final directory."""

    if not directory.parent.exists():
        return ()
    return tuple(directory.parent.glob(f".{directory.name}.*.staging"))


def ensure_destination_available(directory: Path) -> None:
    """Refuse final state and residual staging rather than guessing recovery intent."""

    if os.path.lexists(directory):
        raise FileExistsError(f"official artifact directory already exists: {directory}")
    residual = staging_directories(directory)
    if residual:
        raise RuntimeError(
            "residual artifact staging requires inspection before retry: "
            + ", ".join(map(str, residual))
        )


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_published_artifacts(
    directory: Path, filenames: Sequence[str]
) -> tuple[Path, ...]:
    destinations = tuple(directory / filename for filename in filenames)
    actual = {path.name for path in directory.iterdir()}
    if actual != set(filenames) or any(
        not destination.is_file() for destination in destinations
    ):
        raise RuntimeError("published artifact directory is not the exact registered set")
    return destinations


def _descriptor_metadata(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        file_stat.st_dev,
        file_stat.st_ino,
        file_stat.st_mode,
        file_stat.st_size,
        file_stat.st_mtime_ns,
        file_stat.st_ctime_ns,
    )


def _descriptor_sha256(file_descriptor: int) -> str:
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while chunk := os.read(file_descriptor, 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def verify_publication(receipt: PublicationReceipt) -> tuple[Path, ...]:
    """Authenticate one stable, descriptor-backed snapshot of the publication."""

    identity = (receipt.st_dev, receipt.st_ino)
    contents_message = "published artifact contents differ from publication receipt"
    replacement_message = (
        "published artifact publication replaced/incompatible with receipt; "
        f"replaced by a foreign object: {receipt.directory}"
    )
    try:
        directory_descriptor = os.open(
            receipt.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
    except OSError as error:
        raise RuntimeError(replacement_message) from error
    file_descriptors: list[int] = []
    opened_files: list[tuple[str, str, int, tuple[int, ...]]] = []
    try:
        opened_stat = os.fstat(directory_descriptor)
        opened_identity = (opened_stat.st_dev, opened_stat.st_ino)
        if not stat.S_ISDIR(opened_stat.st_mode) or opened_identity != identity:
            raise RuntimeError(replacement_message)

        expected_names = tuple(filename for filename, _ in receipt.file_hashes)
        actual_names = os.listdir(directory_descriptor)
        if len(actual_names) != len(expected_names) or set(actual_names) != set(
            expected_names
        ):
            raise RuntimeError(contents_message)

        for filename, expected_hash in receipt.file_hashes:
            try:
                file_descriptor = os.open(
                    filename,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
            except OSError as error:
                raise RuntimeError(contents_message) from error
            file_descriptors.append(file_descriptor)
            file_stat = os.fstat(file_descriptor)
            metadata = _descriptor_metadata(file_stat)
            opened_files.append(
                (filename, expected_hash, file_descriptor, metadata)
            )
            if not stat.S_ISREG(file_stat.st_mode):
                raise RuntimeError(contents_message)

        first_hashes: dict[str, str] = {}
        for filename, expected_hash, file_descriptor, _ in opened_files:
            actual_hash = _descriptor_sha256(file_descriptor)
            if actual_hash != expected_hash:
                raise RuntimeError(contents_message)
            first_hashes[filename] = actual_hash

        for filename, expected_hash, file_descriptor, _ in opened_files:
            actual_hash = _descriptor_sha256(file_descriptor)
            if actual_hash != expected_hash or actual_hash != first_hashes[filename]:
                raise RuntimeError(contents_message)

        for _, _, file_descriptor, initial_metadata in opened_files:
            final_metadata = _descriptor_metadata(os.fstat(file_descriptor))
            if final_metadata != initial_metadata:
                raise RuntimeError(contents_message)

        try:
            final_descriptor = os.open(
                receipt.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except OSError as error:
            raise RuntimeError(replacement_message) from error
        try:
            final_opened_stat = os.fstat(final_descriptor)
            if (
                not stat.S_ISDIR(final_opened_stat.st_mode)
                or (final_opened_stat.st_dev, final_opened_stat.st_ino) != identity
                or opened_identity != identity
            ):
                raise RuntimeError(replacement_message)
            final_names = os.listdir(final_descriptor)
            if len(final_names) != len(expected_names) or set(final_names) != set(
                expected_names
            ):
                raise RuntimeError(contents_message)
            for filename, expected_hash, _, initial_metadata in opened_files:
                try:
                    named_descriptor = os.open(
                        filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=final_descriptor
                    )
                except OSError as error:
                    raise RuntimeError(contents_message) from error
                try:
                    named_stat = os.fstat(named_descriptor)
                    if (
                        not stat.S_ISREG(named_stat.st_mode)
                        or _descriptor_metadata(named_stat) != initial_metadata
                        or _descriptor_sha256(named_descriptor) != expected_hash
                        or _descriptor_metadata(os.fstat(named_descriptor)) != initial_metadata
                    ):
                        raise RuntimeError(contents_message)
                finally:
                    os.close(named_descriptor)
        finally:
            os.close(final_descriptor)
        # A prior name may be edited in place while a later name is hashed.
        # Recheck every retained inode and every name after the last hash.
        try:
            terminal_descriptor = os.open(
                receipt.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
        except OSError as error:
            raise RuntimeError(replacement_message) from error
        try:
            terminal_stat = os.fstat(terminal_descriptor)
            if (
                not stat.S_ISDIR(terminal_stat.st_mode)
                or (terminal_stat.st_dev, terminal_stat.st_ino) != identity
            ):
                raise RuntimeError(replacement_message)
            terminal_names = os.listdir(terminal_descriptor)
            if len(terminal_names) != len(expected_names) or set(terminal_names) != set(
                expected_names
            ):
                raise RuntimeError(contents_message)
            for filename, _, file_descriptor, initial_metadata in opened_files:
                if _descriptor_metadata(os.fstat(file_descriptor)) != initial_metadata:
                    raise RuntimeError(contents_message)
                try:
                    named_descriptor = os.open(
                        filename, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=terminal_descriptor
                    )
                except OSError as error:
                    raise RuntimeError(contents_message) from error
                try:
                    named_stat = os.fstat(named_descriptor)
                    if (
                        not stat.S_ISREG(named_stat.st_mode)
                        or _descriptor_metadata(named_stat) != initial_metadata
                        or _descriptor_metadata(os.fstat(file_descriptor)) != initial_metadata
                    ):
                        raise RuntimeError(contents_message)
                finally:
                    os.close(named_descriptor)
        finally:
            os.close(terminal_descriptor)
        return receipt.paths
    finally:
        for file_descriptor in reversed(file_descriptors):
            os.close(file_descriptor)
        os.close(directory_descriptor)


def publish_artifact_directory(
    directory: Path,
    documents: Mapping[str, bytes],
    filenames: Sequence[str],
) -> PublicationReceipt:
    """Fsync and publish a complete file set with one sibling-directory rename."""

    ensure_destination_available(directory)
    if set(documents) != set(filenames):
        raise AssertionError("artifact documents differ from the registered file set")

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{directory.name}.",
            suffix=".staging",
            dir=directory.parent,
        )
    )
    published = False
    staging_descriptor: int | None = None
    try:
        for filename in filenames:
            with (staging / filename).open("wb") as handle:
                handle.write(documents[filename])
                handle.flush()
                os.fsync(handle.fileno())
        fsync_directory(staging)
        staging_descriptor = os.open(
            staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        source_stat = os.fstat(staging_descriptor)
        receipt = PublicationReceipt(
            directory=directory,
            st_dev=source_stat.st_dev,
            st_ino=source_stat.st_ino,
            file_hashes=tuple(
                (filename, hashlib.sha256(documents[filename]).hexdigest())
                for filename in filenames
            ),
        )
        _rename_noreplace(staging, directory)
        published = True
        try:
            opened_stat = os.fstat(staging_descriptor)
            if (opened_stat.st_dev, opened_stat.st_ino) != (
                source_stat.st_dev,
                source_stat.st_ino,
            ):
                raise RuntimeError(
                    "published staging identity changed during publication"
                )
            fsync_directory(directory.parent)
            verify_publication(receipt)
        except BaseException as error:
            raise PublishedArtifactError(receipt, error) from error
    finally:
        if staging_descriptor is not None:
            os.close(staging_descriptor)
        if not published and staging.exists():
            shutil.rmtree(staging)
    return receipt
