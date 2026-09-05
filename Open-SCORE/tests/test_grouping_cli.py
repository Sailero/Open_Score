"""The experiment runner resumes completed cells without retraining them."""
from pathlib import Path

from open_score.grouping.cli import main
from open_score.grouping.storage import read_jsonl, sha256


def test_comparison_resume_preserves_completed_models_and_evaluation(tmp_path):
    args = ['compare', '--profile', 'smoke', '--methods', 'selective', 'full',
            '--steps', '16', '--train-seconds', '60', '--eval-episodes', '1',
            '--diagnostic-states', '1', '--diagnostic-rollouts', '1',
            '--diagnostic-seconds', '60', '--output', str(tmp_path)]
    main(args)
    models = list(tmp_path.rglob('latest.pt'))
    evaluations = list(tmp_path.rglob('evaluation.jsonl'))
    assert len(models) == 2 and len(evaluations) == 1
    before = {str(path): sha256(path) for path in models + evaluations}
    assert len(read_jsonl(evaluations[0])) == 2
    main(args + ['--resume'])
    assert before == {path: sha256(Path(path)) for path in before}
