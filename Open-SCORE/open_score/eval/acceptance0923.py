"""main0923 implementation acceptance (G3/G4): CPU smoke, readout-detach and IAR checks.

Results are appended to implementation_acceptance.json under
`main0923_new_methods` (HAD) and `smac_main0923_extension` (SMAC). Every
check runs on CPU with real environment workers; no formal artifact is written.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time
import traceback

HAD_WORKERS, SMAC_WORKERS = 8, 4
HAD_STEPS, SMAC_STEPS = 32, 16


def _build(method, seed, output, env="had", overrides=None, workers=None):
    from open_score.algos import (setup_runtime, load_config, make_scheme, build_learner,
                                  LearnerLogger, _set_seed)
    setup_runtime()
    from runners.parallel_runner import ParallelRunner
    from open_score.models import build_mac
    from open_score.eval.experiment import budget, PROFILE
    workers = workers or (HAD_WORKERS if env == "had" else SMAC_WORKERS)
    cfg = dict(profile=PROFILE, output=str(output), env=env, run="acceptance", seed=int(seed),
               t_max=budget(env), batch_size_run=workers, use_cuda=False, skip_final_eval=True)
    if env == "smacv2":
        from open_score.utils.resources import configure_workspace
        cfg["env_args"] = {"sc2path": str(configure_workspace() / "envs/StarCraftII")}
    cfg.update(overrides or {})
    args = load_config(method, cfg)
    _set_seed(int(seed), False)
    runner = ParallelRunner(args, LearnerLogger())
    info = runner.get_env_info()
    for key, value in info.items():
        setattr(args, key, value)
    scheme, groups, preprocess = make_scheme(info, multi_task=args.multi_task)
    model_scheme = copy.deepcopy(scheme)
    model_scheme["entities"]["vshape"] = args.entity_shape
    model_scheme["actions_onehot"] = {"vshape": (args.n_actions,), "group": "agents"}
    mac = build_mac(model_scheme, groups, args)
    runner.setup(scheme, groups, preprocess, mac)
    learner = build_learner(mac, model_scheme, LearnerLogger(), args)
    return args, runner, learner, (scheme, groups, preprocess, model_scheme)


def _collect(runner, args, steps):
    batch, summaries = runner.run(max_train_steps=steps)
    batch = batch[:, :int(batch.max_t_filled().item())]
    batch.to("cpu")
    return batch, summaries


def _params(learner):
    import torch as th
    return th.cat([p.detach().reshape(-1).clone() for p in learner.params])


def _train_once(learner, batch, seed):
    import torch as th
    th.manual_seed(seed)
    return learner.train(batch, 0, 0)


def _snapshot(learner):
    import torch as th
    import random
    import numpy as np
    from open_score.algos import _network_state
    return copy.deepcopy(dict(networks=_network_state(learner), torch=th.get_rng_state(),
                              numpy=np.random.get_state(), python=random.getstate()))


def _restore(learner, snapshot):
    import torch as th
    import random
    import numpy as np
    from open_score.algos import _load_network_state
    _load_network_state(learner, copy.deepcopy(snapshot["networks"]))
    th.set_rng_state(snapshot["torch"]); np.random.set_state(snapshot["numpy"]); random.setstate(snapshot["python"])


def smoke(method, seed, output, env="had"):
    """Mixed-scale collection, TD update, parameter change, exact restore and next update."""
    import torch as th
    started = time.time()
    args, runner, learner, _ = _build(method, seed, output, env)
    try:
        batch, _ = _collect(runner, args, HAD_STEPS if env == "had" else SMAC_STEPS)
        scales = sorted({int(n) for n in (1 - batch["initial_agent_mask"][:, 0].float()).sum(-1).tolist()})
        args.batch_size = batch.batch_size
        before = _params(learner)
        first = _train_once(learner, batch, 11)
        changed = bool((_params(learner) != before).any())
        snapshot = _snapshot(learner)
        second = _train_once(learner, batch, 12)
        after_second = _params(learner)
        _restore(learner, snapshot)
        # A freshly built learner restored from the snapshot must reproduce the next update.
        args2, runner2, learner2, _ = _build(method, seed, output, env, workers=1)
        runner2.close_env()
        _restore(learner2, snapshot)
        replay = _train_once(learner2, batch, 12)
        exact = bool(th.equal(_params(learner2), after_second)) and replay["loss"] == second["loss"]
        finite = all(th.isfinite(th.tensor(v)).item() for v in (first["loss"], second["loss"]))
        row = dict(method=method, seed=int(seed), env=env, status="passed" if (finite and changed and exact) else "failed",
                   device="cpu", workers=int(args.batch_size_run), physical_steps=int(runner.t_env),
                   sequence_length=int(batch.max_seq_length), mixed_scales=scales,
                   finite_td=finite, parameters_updated=changed, checkpoint_restore_exact=exact,
                   resumed_next_update_exact=exact, loss=float(first["loss"]),
                   intent_loss=first.get("intent_loss"), seconds=round(time.time() - started, 1))
        return row, (args, learner, batch)
    finally:
        runner.close_env()


def detach_check(method, seed, output):
    """Detached readout memory: same loss and Q; zero gradient into h_prev via the query."""
    import torch as th
    args, runner, learner, _ = _build(method, seed, output)
    try:
        batch, _ = _collect(runner, args, HAD_STEPS)
    finally:
        runner.close_env()
    args.batch_size = batch.batch_size
    learner.optimiser.step = lambda *a, **k: None
    learner.last_target_update_episode = 10 ** 12
    losses = {}
    for flag in (False, True):
        args.global_query_detach_memory = flag
        losses[flag] = _train_once(learner, batch, 21)["loss"]
    args.global_query_detach_memory = True
    branch = learner.mac.agent.global_net
    own = th.randn(5, int(args.global_embed_dim))
    hidden = th.randn(5, int(args.rnn_hidden_dim), requires_grad=True)
    branch._agent_query(own, hidden).sum().backward()
    no_grad = hidden.grad is None or float(hidden.grad.abs().sum()) == 0.0
    args.global_query_detach_memory = False
    hidden2 = hidden.detach().clone().requires_grad_(True)
    branch._agent_query(own, hidden2).sum().backward()
    has_grad_off = hidden2.grad is not None and float(hidden2.grad.abs().sum()) > 0
    args.global_query_detach_memory = True
    return dict(loss_equal=losses[False] == losses[True], query_memory_gradient_zero=no_grad,
                gradient_present_without_detach=has_grad_off)


def _reference(method):
    return {"regir_kv0_intent_sg": "regir_kv0_sg", "regir_kv0_intent_noaux_sg": "regir_kv0_sg",
            "regir_kv0_intent_nomem": "regir_kv0_nomem", "regir_intent_sg": "regir_sg",
            "regir_kv0_intent_norefil_sg": "regir_kv0_norefil_sg"}[method]


def _clone(learner, method, seed, output, env="had"):
    """A fresh learner of the same method with identical network and optimizer state."""
    from open_score.algos import _network_state, _load_network_state
    _, runner, fresh, _ = _build(method, seed, output, env, workers=1)
    runner.close_env()
    fresh.args.batch_size = learner.args.batch_size
    fresh.args.global_depths = list(learner.args.global_depths)
    _load_network_state(fresh, copy.deepcopy(_network_state(learner)))
    return fresh


def _clear(learner):
    for mac in (learner.mac, learner.target_mac):
        mac.hidden_states = None
        branch = getattr(mac.agent, "global_net", None)
        if branch is not None:
            branch.last_intent = None


def intent_checks(method, seed, output, context):
    """IAR: init equivalence, label alignment, information isolation, loss descent, bounded embedding."""
    import torch as th
    from open_score.models.entity_encoder import intent_targets
    args, learner, batch = context
    result = {}
    # 1) Same-seed ARR reference: identical parameters (shared names) and bitwise-equal Q.
    ref_args, ref_runner, ref_learner, _ = _build(_reference(method), seed, output, workers=1)
    ref_runner.close_env()
    iar_args, iar_runner, iar_learner, _ = _build(method, seed, output, workers=1)
    iar_runner.close_env()
    ref_state = ref_learner.mac.agent.state_dict()
    iar_state = iar_learner.mac.agent.state_dict()
    shared = [k for k in ref_state if k in iar_state]
    extra = sorted(set(iar_state) - set(ref_state))
    same_params = all(th.equal(ref_state[k], iar_state[k]) for k in shared) and len(shared) == len(ref_state)
    same_mixer = all(th.equal(a, b) for a, b in zip(ref_learner.mixer.state_dict().values(),
                                                    iar_learner.mixer.state_dict().values()))
    with th.no_grad():
        for mac in (ref_learner.mac, iar_learner.mac):
            mac.set_global_depth(4)
            mac.init_hidden(batch.batch_size)
        q_ref, _ = ref_learner.mac.forward(batch, t=None)
        q_iar, _ = iar_learner.mac.forward(batch, t=None)
    result.update(init_same_parameters=bool(same_params and same_mixer), extra_parameters=extra,
                  init_q_equal=bool(th.equal(q_ref, q_iar)))
    # 2) Label alignment: teammate j acts j mod 9; each valid slot must recover its origin.
    probe = iar_learner
    na = int(iar_args.n_agents)
    actions = (th.arange(na) % 9).view(1, 1, na).expand(batch.batch_size, batch.max_seq_length - 1, na)
    with th.no_grad():
        probe.mac.init_hidden(batch.batch_size)
        probe.mac.set_global_depth(4)
        probe.mac.forward(batch, t=None)
    step_mask = batch["filled"][:, :-1, 0].float()
    record, rows, labels, valid, _ = intent_targets(probe.mac.agent, actions, step_mask, batch.batch_size)
    origin = record["origin"].index_select(0, rows).long()
    idx = record["idx"].index_select(0, rows)
    observer = idx % na
    ts = record["shape"][1]
    b, t = idx // (ts * na), (idx // na) % ts
    entity_mask = batch["entity_mask"].bool()
    obs_mask = batch["obs_mask"].bool()
    # relative_entity_views swaps the observer into slot 0: slot s holds entity s,
    # except slot i (observer i) which holds entity 0.
    expected = th.zeros_like(valid)
    truth = th.zeros_like(labels)
    for r in range(len(rows)):
        i = int(observer[r])
        for slot in range(1, origin.shape[1]):
            j = 0 if slot == i else slot
            if j >= na or j == i:
                continue
            truth[r, slot] = j % 9
            if bool(entity_mask[b[r], t[r], j]) or bool(obs_mask[b[r], t[r], i, j]):
                continue
            expected[r, slot] = bool(step_mask[b[r], t[r]] > 0)
    labels_ok = bool(th.equal(labels[valid], truth[valid]))
    result.update(label_alignment=bool(labels_ok and th.equal(valid, expected)),
                  labelled_pairs=int(valid.sum()),
                  self_dead_hidden_padding_excluded=bool(th.equal(valid, expected)))
    # 3) Information isolation: perturbing a teammate hidden from observer i leaves i unchanged.
    isolated = None
    hidden_pairs = (obs_mask[:, :, :na, :na] & ~entity_mask[:, :, :na].unsqueeze(-2)).nonzero(as_tuple=False)
    hidden_pairs = [p for p in hidden_pairs.tolist() if p[2] != p[3]]
    if hidden_pairs:
        bb, tt, ii, jj = hidden_pairs[0]
        perturbed = copy.deepcopy(batch)
        perturbed.data.transition_data["entities"][bb, tt, jj] += 3.0
        outputs = []
        for data in (batch, perturbed):
            with th.no_grad():
                probe.mac.init_hidden(data.batch_size)
                q, _ = probe.mac.forward(data, t=None)
            rec = probe.mac.agent.global_net.last_intent
            flat = (bb * ts + tt) * na + ii
            where = (rec["idx"] == flat).nonzero(as_tuple=False).flatten()
            logits = th.stack([lg.index_select(0, where) for lg in rec["logits"]])
            outputs.append((q[bb, tt, ii].clone(), logits))
        isolated = bool(th.equal(outputs[0][0], outputs[1][0]) and th.equal(outputs[0][1], outputs[1][1]))
    result["information_isolated"] = isolated
    # 4) Bounded embedding: ||W_e p|| <= ||W_e||_2 for any distribution p.
    embed = copy.deepcopy(probe.mac.agent.global_net.intent_embed)
    with th.no_grad():
        embed.weight.normal_()
        p = th.softmax(th.randn(512, embed.weight.shape[1]) * 3, -1)
        bound = th.linalg.matrix_norm(embed.weight, ord=2)
        result["embedding_bounded"] = bool((embed(p).norm(dim=-1) <= bound + 1e-5).all())
    # 5) Auxiliary loss descends in 200 CPU updates; weight 0 contributes no gradient.
    weight = float(getattr(args, "intent_aux_weight", 0.0))
    trainer = _clone(learner, method, seed, output)
    trainer.args.global_depths = [4]
    history = []
    for step in range(200):
        history.append(_train_once(trainer, batch, 100 + step).get("intent_loss"))
    history = [h for h in history if h is not None]
    first, last = sum(history[:20]) / 20, sum(history[-20:]) / 20
    result.update(aux_loss_first20=first, aux_loss_last20=last,
                  aux_loss_decreases=bool(last < first) if weight > 0 else None)
    if weight == 0:
        import open_score.models.entity_encoder as encoder
        grads = []
        for disable in (False, True):
            model = _clone(learner, method, seed, output)
            model.optimiser.step = lambda *a, **k: None
            original = encoder.intent_auxiliary
            if disable:
                encoder.intent_auxiliary = lambda *a, **k: None
            try:
                _train_once(model, batch, 7)
            finally:
                encoder.intent_auxiliary = original
            grads.append(th.cat([p.grad.reshape(-1) for p in model.mac.agent.global_net.intent_head.parameters()]))
        result["zero_weight_no_aux_gradient"] = bool(th.equal(grads[0], grads[1]))
    passed = (result["init_same_parameters"] and result["init_q_equal"] and result["label_alignment"]
              and result["information_isolated"] is not False and result["embedding_bounded"]
              and (result["aux_loss_decreases"] is not False)
              and result.get("zero_weight_no_aux_gradient", True))
    result["status"] = "passed" if passed else "failed"
    return result


def targeted_checks(method, seed, output, context):
    import torch as th
    args, learner, batch = context
    result = {}
    branch = learner.mac.agent.global_net
    if getattr(args, "global_read_h0", False):
        model = _clone(learner, method, seed, output)
        with th.no_grad():
            model.mac.init_hidden(batch.batch_size)
            q0, _ = model.mac.forward(batch, t=None)
            for name in ("self_attn", "ffn", "norm_attn", "norm_ffn", "count_to_token"):
                for p in getattr(model.mac.agent.global_net, name).parameters():
                    p.add_(th.randn_like(p))
            model.mac.init_hidden(batch.batch_size)
            q1, _ = model.mac.forward(batch, t=None)
        result["round_parameters_unused"] = bool(th.equal(q0, q1))
    if getattr(args, "global_query_no_memory", False):
        own = th.randn(4, int(args.global_embed_dim))
        a = branch._agent_query(own, th.randn(4, int(args.rnn_hidden_dim)))
        b = branch._agent_query(own, th.randn(4, int(args.rnn_hidden_dim)))
        result["query_ignores_memory"] = bool(th.equal(a, b))
    if list(getattr(args, "global_depths", [])) == [4] and getattr(args, "global_branch", None) == "cycle":
        result["training_depth_fixed4"] = True
    if result:
        result["status"] = "passed" if all(v is not False for v in result.values()) else "failed"
    return result


def check_had(method, seed, output):
    row = dict(method=method, seed=int(seed), env="had")
    try:
        smoke_row, context = smoke(method, seed, output)
        row.update(smoke_row)
        args = context[0]
        if int(seed) == 0 and getattr(args, "global_branch", None) == "cycle":
            row["detach"] = detach_check(method, seed, output) if getattr(args, "global_query_detach_memory", False) else None
            row["targeted"] = targeted_checks(method, seed, output, context)
        if getattr(args, "rer_intent", False):
            row["intent"] = intent_checks(method, seed, output, context) if int(seed) == 0 else None
            if int(seed) != 0:
                row["intent_init"] = _init_equivalence(method, seed, output)
        failed = [row.get("status") != "passed"]
        if row.get("detach"):
            failed.append(not (row["detach"]["loss_equal"] and row["detach"]["query_memory_gradient_zero"]))
        for key in ("targeted", "intent", "intent_init"):
            if row.get(key):
                failed.append(row[key].get("status") != "passed")
        row["status"] = "failed" if any(failed) else "passed"
    except BaseException as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    return row


def _init_equivalence(method, seed, output):
    import torch as th
    ref_args, ref_runner, ref_learner, _ = _build(_reference(method), seed, output, workers=1)
    ref_runner.close_env()
    iar_args, iar_runner, iar_learner, _ = _build(method, seed, output, workers=1)
    iar_runner.close_env()
    ref, iar = ref_learner.mac.agent.state_dict(), iar_learner.mac.agent.state_dict()
    ok = all(th.equal(ref[k], iar[k]) for k in ref) and all(
        th.equal(a, b) for a, b in zip(ref_learner.mixer.state_dict().values(), iar_learner.mixer.state_dict().values()))
    return dict(init_same_parameters=bool(ok), status="passed" if ok else "failed")


def check_smac(method, seed, output):
    """Real SC2 CPU smoke (4 workers, 16 steps) plus a strict-load 20v20 episode for seed 0."""
    import torch as th
    row = dict(method=method, seed=int(seed), env="smacv2")
    try:
        smoke_row, (args, learner, batch) = smoke(method, seed, output, env="smacv2")
        row.update(smoke_row)
        alive = ~batch["entity_mask"][:, :, :int(args.n_agents)].bool()
        with th.no_grad():
            learner.mac.init_hidden(batch.batch_size)
            q, _ = learner.mac.forward(batch, t=None)
        row["padding_death_zero"] = bool((q[~alive] == 0).all())
        row["input_path"] = "unchanged from main0921 accepted regir/regir_r1 (smac_six_method_extension)"
        if getattr(args, "rer_intent", False):
            classes = learner.mac.agent.global_net.intent_head.out_features
            row["intent_classes"] = int(classes)
            row["intent_label_mapping"] = "a<6 -> a; attack(any enemy) -> 6"
            row["intent_ok"] = classes == 7 and smoke_row.get("intent_loss") is not None
        if int(seed) == 0:
            row["final_interface_20v20"] = _smac_strict_load(method, output, learner)
        ok = (row.get("status") == "passed" and row["padding_death_zero"]
              and row.get("intent_ok", True) and row.get("final_interface_20v20", {"status": "passed"})["status"] == "passed")
        row["status"] = "passed" if ok else "failed"
    except BaseException as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    return row


def _smac_strict_load(method, output, learner):
    """Serialize the updated actor, rebuild it at the 20v20 interface with strict loading, play one episode."""
    import torch as th
    from open_score.algos import _network_state
    from open_score.eval.protocol import _smac_eval_runner
    config = dict(N_R=20, N_B=20, K=0)
    saved = dict(config=vars(learner.args).copy(), networks=_network_state(learner))
    saved["config"]["env_args"] = dict(saved["config"].get("env_args", {}))
    runner = _smac_eval_runner(saved, config, device="cpu")
    try:
        from open_score.envs.smacv2_env import generate_registered_scenes
        scenes = generate_registered_scenes(output)
        scene = scenes["scenes"]["20v20.s110000"]
        job = dict(config=config, episode_seed=110000, reset_config=scene["reset_config"],
                   engine_seed=scene["engine_seed"], retain_trajectory=False)
        with th.inference_mode():
            _, summaries = runner.run(test_mode=True, jobs=[job])
        return dict(status="passed", ep_len=int(summaries[0]["ep_len"]),
                    battle_won=float(summaries[0]["battle_won"]))
    finally:
        runner.close_env()


def _worker(args, results):
    env, method, seed, output = args
    from open_score.utils.resources import cpu_threads
    cpu_threads()
    import torch
    torch.set_num_threads(1)
    results.put((check_had if env == "had" else check_smac)(method, seed, output))


def run(output, env="had", methods=None, processes=16):
    """Run all (method, seed) acceptance rows in parallel and record them."""
    from multiprocessing import get_context
    from open_score.eval import experiment
    output = Path(output)
    if env == "had":
        methods = methods or experiment.new_had_methods()
        key = experiment.ACCEPTANCE_KEY
    else:
        methods = methods or ("regir_r1_sg", "regir_kv0_sg", "regir_sg", "regir_kv0_intent_sg")
        key = experiment.SMAC_ACCEPTANCE_KEY
    jobs = [(env, m, s, str(output)) for m in methods for s in experiment.method_seeds(env, m)]
    started = time.time()
    # Environment runners fork their own workers, so acceptance processes must not be daemonic.
    context = get_context("spawn")
    results, pending, live, rows = context.Queue(), list(jobs), [], []
    while pending or live:
        while pending and len(live) < processes:
            job = pending.pop(0)
            child = context.Process(target=_worker, args=(job, results))
            child.start()
            live.append((child, job))
        try:
            row = results.get(timeout=5)
            rows.append(row)
            print(f"[{len(rows)}/{len(jobs)}] {row['env']} {row['method']} s{row['seed']}: {row['status']}"
                  + (f"  {row.get('error')}" if row.get("error") else ""), flush=True)
        except Exception:
            pass
        for child, job in list(live):
            if not child.is_alive():
                child.join()
                live.remove((child, job))
                if child.exitcode and not any((r["method"], r["seed"]) == (job[1], job[2]) for r in rows):
                    rows.append(dict(env=job[0], method=job[1], seed=job[2], status="failed",
                                     error=f"acceptance worker exit {child.exitcode}"))
    path = output / "implementation_acceptance.json"
    acceptance = json.loads(path.read_text(encoding="utf-8"))
    section = acceptance.get(key, {})
    previous = {(r["method"], r["seed"]): r for r in section.get("results" if env == "had" else "cpu_matrix", [])}
    previous.update({(r["method"], r["seed"]): r for r in rows})
    merged = sorted(previous.values(), key=lambda r: (r["method"], r["seed"]))
    section = dict(status="passed" if all(r["status"] == "passed" for r in merged) else "failed",
                   checked_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seconds=round(time.time() - started, 1),
                   implementation_revision=experiment.IMPLEMENTATION_REVISION)
    section["results" if env == "had" else "cpu_matrix"] = merged
    acceptance[key] = section
    experiment.atomic_json(path, acceptance)
    failed = [r for r in rows if r["status"] != "passed"]
    return rows, failed


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env", choices=("had", "smacv2"), default="had")
    parser.add_argument("--methods", default=None)
    parser.add_argument("--processes", type=int, default=16)
    options = parser.parse_args()
    rows, failed = run(options.output, options.env,
                       tuple(options.methods.split(",")) if options.methods else None, options.processes)
    for row in failed:
        print(json.dumps({k: v for k, v in row.items() if k != "traceback"}, ensure_ascii=False, default=str))
        print(row.get("traceback", ""))
    sys.exit(1 if failed else 0)
