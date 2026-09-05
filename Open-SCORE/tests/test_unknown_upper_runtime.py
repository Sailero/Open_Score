"""Physical integration checks; temporary units are never formal evidence."""
from pathlib import Path
import copy
import sys

import numpy as np
import pytest
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_stage123_unknown_upper as pipeline
from open_score.stage3.unknown_upper.evaluation import ablation_registry, execute_episode, formal_registry
from open_score.stage3.unknown_upper.planner import ABLATION_METHODS, METHODS, MCP_METHODS, Commander
from open_score.stage3.unknown_upper.training import load_qom, train_qom


def configuration():
    return yaml.safe_load((PROJECT / "configs/stage123_unknown_upper.yaml").read_text(encoding="utf8"))


def test_formal_protocol_and_matched_paper_controls():
    config = configuration()
    pipeline.validate_protocol(config)
    core, controls = formal_registry(config), ablation_registry(config)
    assert len(core) == 1920 and len(controls) == 480
    assert {c["seed"] for c in controls} == {c["seed"] for c in core}
    assert not {c["id"] for c in controls} & {c["id"] for c in core}
    altered = copy.deepcopy(config)
    altered["physical"]["command_interval"] = 7
    with pytest.raises(ValueError, match="clock"):
        pipeline.validate_protocol(altered)


def test_complete_result_merge_includes_controls_and_both_lower_policies(tmp_path,monkeypatch):
    from open_score.stage3.unknown_upper import analysis
    config = configuration()
    config["acceptance"]["bootstrap_replicates"] = 100
    core, controls = formal_registry(config), ablation_registry(config)
    def result(cell):
        return {"cell":cell,"red_win":int(cell["method"] not in {"legacy_idb","event_risk_idb"}),
                "identity_valid":True,"fixed_targets":True,"infeasible":False,"steps":12,
                "events":[{"planning_seconds":.1,"event_index":1,"truth_targets":[0],"predicted_targets":[0],"forecast_oracle_accuracy_ceiling":1.0}],
                "filtered_assignments":[{"event_index":1,"truth_targets":[0],"predicted_targets":[0]}],
                "surprise":[{"step":11,"nll":float(cell["upper"]=="feint_switch")}]}
    # Test bookkeeping on synthetic fixtures, never physics or formal results.
    monkeypatch.setattr(analysis,"read_unit",result)
    root = tmp_path/"evidence/stage3/unknown_upper"
    summary = analysis.summarize(core,config,root,{"hard_checks_passed":True,"snapshots":[]},controls)
    assert summary["episodes"] == 1920 and summary["ablation_episodes"] == 480
    assert len(summary["methods"]) == 10
    assert set(summary["oracle_effect_by_lower"]) == {"rush","split_rush"}
    assert summary["matched_controls"]["learned_vs_finite_belief"]["pairs"] == 240
    assert summary["all_passed"]


def test_physical_workers_training_reload_all_commanders_and_resume(tmp_path):
    config = configuration()
    frozen = PROJECT / config["output_dir"]
    if not (frozen / config["artifacts"]["stage1"]).exists():
        pytest.skip("Local frozen models are not distributed through Git")
    config["training_device"] = "cpu"
    config["training"].update(epochs=1, batch_size=2)
    config["physical"]["max_steps"] = 6
    for name in ["stage1", "stage2", "stage2_data"]:
        config["artifacts"][name] = str(frozen / config["artifacts"][name])
    scenario = {"label":"integration_only", "red":2, "blue":2, "targets":2}
    cells = [{"scenario":scenario, "lower":lower, "upper":"balanced", "probe":"balanced",
              "seed":200+i*2+j, "id":f"integration_{i}_{j}", "split":split}
             for i, split in enumerate(["train", "validation", "test"])
             for j, lower in enumerate(["rush", "split_rush"])]
    status_rows = []
    status = lambda phase, complete, expected, **extra: status_rows.append((phase, complete, expected, extra))
    paths, entries = pipeline.run_units(cells, "collect_qom", {"integration":1}, config, tmp_path, 2,
                                       lambda text: None, status)
    assert len(paths) == 6 and all(p.exists() for p in paths)
    initial_hashes = [entry["file_sha256"] for entry in entries]
    _, reused = pipeline.run_units(cells, "collect_qom", {"integration":1}, config, tmp_path, 2,
                                  lambda text: None, status)
    assert status_rows[-1][3]["reused"] == 6
    assert initial_hashes == [entry["file_sha256"] for entry in reused]
    pipeline.initialize(config, tmp_path)
    _, predictor, stage1, device, _ = pipeline._WORKER
    model_dir = tmp_path / "test_model"
    metrics = train_qom(paths, config, model_dir, predictor, lambda text: None)
    assert metrics["heldout_online"]["episodes"] == 2
    assert metrics["heldout_online"]["identity_predictions"] > 0
    assert metrics["heldout_online"]["labels_used_for_scoring_only"]
    model, factory = load_qom(model_dir / "qom.pt", device)
    assert not model.training
    commander = Commander(predictor, stage1, device, config["planning"], factory)
    for index, method in enumerate(METHODS + ABLATION_METHODS):
        cell = {"scenario":scenario, "lower":"split_rush", "upper":"balanced", "method":method,
                "seed":320+index, "id":f"integration_{method}"}
        result = execute_episode(cell, config, predictor, stage1, device, commander)
        assert result["fixed_targets"] and result["identity_valid"] and not result["infeasible"]
        assert result["steps"] == 6
        assert len(result["events"]) == 2
        assert all(e["puct_simulations"] == (32 if method in MCP_METHODS else 0) for e in result["events"])
        assert all(np.isclose(sum(e["posterior"]), 1) for e in result["events"])
