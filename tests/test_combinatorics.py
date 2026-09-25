from loto.combinatorics import jackpot_outcome_space, number_combination_count


def test_loto_outcome_space():
    assert number_combination_count() == 1_906_884
    assert jackpot_outcome_space() == 19_068_840