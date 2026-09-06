import numpy as np

from open_score.research_v5.diagnostics import score_audit


def test_independent_ranking_keeps_ties_and_absolute_bce_calibration_separate():
    outcomes = [[0,0,0,0],[0,1,0,1],[0,1,0,1],[1,1,1,1]]
    result = score_audit([0,.5,.5,1.],outcomes,0,[0,.5,.5,1.])
    assert result['non_tie_pairs']==5
    assert result['ranking_accuracy']==1
    assert result['paired_advantage_mse']==0
    assert result['probability_brier']==0
    assert result['probability_ece']==0
    tied = score_audit([0,0,0,0],outcomes,0)
    assert tied['ranking_accuracy']==.5
    assert tied['probability_brier'] is None


def test_all_zero_candidates_have_no_manufactured_ranking():
    result = score_audit([0,.2,-.2],np.zeros((3,4)),0)
    assert result['non_tie_pairs']==0
    assert result['ranking_accuracy'] is None
    assert result['paired_advantage_mse'] > 0
