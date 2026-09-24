"""Shared immutable observation-provenance checks for evaluation models."""

from __future__ import annotations

from datetime import date

from ..walk_forward import DrawObservation, History

ObservationFingerprint = tuple[date, int, tuple[int, int, int, int, int]]


def observation_fingerprint(
    observation: DrawObservation,
) -> ObservationFingerprint:
    return observation.draw_date, observation.original_index, observation.numbers


def validate_training_provenance(
    provenance: object,
    *,
    expected_size: int,
    fitted_through_date: date,
    fitted_through_original_index: int,
) -> tuple[ObservationFingerprint, ...]:
    """Validate complete ordered training identities and the fitted boundary."""

    if type(provenance) is not tuple:
        raise TypeError("state training_provenance must be an immutable tuple")
    if len(provenance) != expected_size:
        raise ValueError("state training_provenance has invalid dimensions")

    checked: list[ObservationFingerprint] = []
    for fingerprint in provenance:
        if type(fingerprint) is not tuple or len(fingerprint) != 3:
            raise ValueError("state training_provenance contains a malformed fingerprint")
        draw_date, original_index, numbers = fingerprint
        if type(numbers) is not tuple:
            raise TypeError(
                "state training_provenance numbers must be an immutable tuple"
            )
        try:
            observation = DrawObservation(draw_date, original_index, numbers)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"state training_provenance contains an invalid fingerprint: {exc}"
            ) from exc
        checked.append(observation_fingerprint(observation))

    result = tuple(checked)
    if any(left[0] >= right[0] for left, right in zip(result, result[1:])):
        raise ValueError("state training_provenance dates must be strictly increasing")
    indices = tuple(fingerprint[1] for fingerprint in result)
    if len(indices) != len(set(indices)):
        raise ValueError("state training_provenance original indices must be unique")
    if result[-1][:2] != (
        fitted_through_date,
        fitted_through_original_index,
    ):
        raise ValueError(
            "state training_provenance is inconsistent with fitted_through fields"
        )
    return result


def validate_visible_training_overlap(
    history: History,
    *,
    training_provenance: tuple[ObservationFingerprint, ...],
    fitted_through_date: date,
) -> None:
    """Require exact shared identities while allowing complete bounded eviction."""

    training_by_index = {
        fingerprint[1]: fingerprint for fingerprint in training_provenance
    }
    visible = tuple(observation_fingerprint(observation) for observation in history)
    for fingerprint in visible:
        training_fingerprint = training_by_index.get(fingerprint[1])
        if training_fingerprint is not None and training_fingerprint != fingerprint:
            raise ValueError("state training provenance diverges from visible history")

    if fitted_through_date > history[-1].draw_date:
        raise ValueError("state fitted observation is later than visible history")
    if fitted_through_date < history[0].draw_date:
        return

    training_first_date = training_provenance[0][0]
    training_overlap = tuple(
        fingerprint
        for fingerprint in training_provenance
        if history[0].draw_date <= fingerprint[0] <= history[-1].draw_date
    )
    visible_overlap = tuple(
        fingerprint
        for fingerprint in visible
        if training_first_date <= fingerprint[0] <= fitted_through_date
    )
    if training_overlap != visible_overlap:
        raise ValueError("state training provenance diverges from visible history")


__all__ = [
    "ObservationFingerprint",
    "observation_fingerprint",
    "validate_training_provenance",
    "validate_visible_training_overlap",
]
