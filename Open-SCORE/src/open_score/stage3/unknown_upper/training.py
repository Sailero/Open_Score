"""Fit QOM on fixed-type episodes and score held-out causal prediction."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch
from .belief import OpponentBelief
from .domain import IdentityUpperAction, PublicState, seed_for
from .policies import RiskProxy
from .qom import QuantizedOpponentModel,QOMRuntime,collate,loss
from .storage import atomic_json,file_hash,read_unit


def evaluate_online(episodes,model,device,prior,thresholds,predictor,seed,log):
    """Held-out prediction is causal; true commands are used only for scoring."""
    correct,total,tp,fp,fn,transitions = 0,0,0,0,0,0
    filter_correct,filter_total = 0,0
    for index,episode in enumerate(episodes):
        belief = OpponentBelief("qom",QOMRuntime(model,device,prior,thresholds))
        commands = {row["public"]["step"]:row for row in episode["commands"]}
        states = [PublicState.from_dict(row) for row in episode["frames"]]
        for before,after in zip(states,states[1:]):
            if before.step in commands:
                belief.prepare(before,RiskProxy(before,predictor),np.random.default_rng(seed_for("test-inference",seed,episode["cell"]["seed"],before.step)))
                row = commands[before.step]
                current_action = IdentityUpperAction.from_dict(row["blue_action"])
                current_event = row["event_index"]
                if row["event_index"] >= 1:
                    action = current_action
                    assignments = action.assignment()
                    ids = before.ids("blue")
                    target_index = {t.id:i for i,t in enumerate(before.targets)}
                    labels = np.asarray([target_index[assignments[i]] for i in ids])
                    prediction = belief.targets(before).argmax(-1)
                    correct += int(np.sum(prediction==labels));total += len(labels)
                    membership = {i:g for g,(_,members) in enumerate(action.groups) for i in members}
                    same = np.asarray([[membership[i]==membership[j] for j in ids] for i in ids])
                    affinity = (belief.event_distributions[1]*belief.posterior[:,None,None]).sum(0)
                    predicted_same = (affinity>=.5) & (prediction[:,None]==prediction[None,:])
                    mask = np.triu(np.ones(same.shape,bool),1)
                    tp += int(np.sum(mask & same & predicted_same))
                    fp += int(np.sum(mask & ~same & predicted_same))
                    fn += int(np.sum(mask & same & ~predicted_same))
            nll = belief.update(before,after)
            if current_event >= 1:
                assignments = current_action.assignment()
                target_index = {t.id:i for i,t in enumerate(after.targets)}
                labels = np.asarray([target_index[assignments[i]] for i in after.ids("blue")])
                prediction = belief.current_targets(after).argmax(-1)
                filter_correct += int(np.sum(prediction==labels));filter_total += len(labels)
            transitions += 1
        if (index+1)%30==0:
            log(f"QOM held-out online evaluation {index+1}/{len(episodes)}")
    return {"episodes":len(episodes),"next_action_forecast_micro_f1":correct/total if total else None,
            "current_action_filter_micro_f1_after_second_event":filter_correct/filter_total if filter_total else None,
            "identity_predictions":total,"same_group_pair_f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "transitions":transitions,"labels_used_for_scoring_only":True,
            "selection":"held-out seeds; checkpoint and thresholds fixed on train/validation"}


def train_qom(paths,config,root,predictor,log):
    episodes = [read_unit(path) for path in paths]
    if any(ep is None for ep in episodes):
        raise ValueError("Training unit failed its content hash")
    sets = {split:[ep for ep in episodes if ep["cell"]["split"]==split] for split in ["train","validation","test"]}
    if any(not episodes for episodes in sets.values()):
        raise ValueError("Train, validation and test episode splits must all be nonempty")
    root.mkdir(parents=True,exist_ok=True)
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    requested = config["training_device"]
    device = torch.device(requested if requested!="cuda" or torch.cuda.is_available() else "cpu")
    cfg = config["training"]
    model = QuantizedOpponentModel(cfg["codebook"],cfg["latent"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
    rng = np.random.default_rng(seed)
    history,best,best_state = [],float("inf"),None
    for epoch in range(cfg["epochs"]):
        model.train()
        batches = rng.permutation(len(sets["train"]))
        total,usage,embeddings = [],np.zeros(model.codes,int),[]
        for start in range(0,len(batches),cfg["batch_size"]):
            selected = [sets["train"][i] for i in batches[start:start+cfg["batch_size"]]]
            batch = collate(selected,device)
            optimizer.zero_grad(set_to_none=True)
            value,parts,codes,z = loss(model,batch)
            if not torch.isfinite(value):
                raise RuntimeError("Non-finite QOM training loss")
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            total.append(float(value.detach()))
            usage += np.bincount(codes.detach().cpu().numpy(),minlength=model.codes)
            embeddings.append(z.detach().cpu())
        model.eval()
        validation = []
        with torch.inference_mode():
            for start in range(0,len(sets["validation"]),cfg["batch_size"]):
                batch = collate(sets["validation"][start:start+cfg["batch_size"]],device)
                _,parts,_,_ = loss(model,batch)
                validation.append(parts["target_loss"]+.25*parts["group_loss"])
        score = float(np.mean(validation))
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(total)),"validation_prediction_loss":score,"code_usage":usage.tolist()})
        if score < best:
            best,best_state = score,copy.deepcopy(model.state_dict())
        # Revive unused codes using train embeddings only, never held-out types.
        if epoch < cfg["epochs"]//2 and (usage==0).any():
            values = torch.cat(embeddings).to(device)
            with torch.no_grad():
                for k in np.flatnonzero(usage==0):
                    model.codebook.weight[k].copy_(values[int(rng.integers(len(values)))]+.01*torch.randn(model.latent,device=device))
        log(f"QOM epoch {epoch+1}/{cfg['epochs']}: validation={score:.4f}, active_codes={int((usage>0).sum())}")
        atomic_json(root/"training_history.json",history)
    model.load_state_dict(best_state)
    model.eval()
    usage = np.ones(model.codes,float) # Dirichlet smoothing on training only.
    with torch.inference_mode():
        for start in range(0,len(sets["train"]),cfg["batch_size"]):
            _,_,_,indices,_ = model(collate(sets["train"][start:start+cfg["batch_size"]],device))
            usage += np.bincount(indices.cpu().numpy(),minlength=model.codes)
    prior = usage/usage.sum()
    # The fixed-type closed-set protocol has no novelty bucket or calibration.
    runtime_device = predictor.device
    model = model.to(runtime_device)
    thresholds = {}
    heldout = evaluate_online(sets["test"],model,runtime_device,prior,thresholds,predictor,seed,log)
    root.mkdir(parents=True,exist_ok=True)
    checkpoint = root/"qom.pt"
    temporary = root/"qom.pt.tmp"
    torch.save({"model":{k:v.cpu() for k,v in model.state_dict().items()},"codes":model.codes,"latent":model.latent,
                "prior":prior.tolist(),"thresholds":thresholds,"seed":seed,"adaptation":"public-motion-VQ-GRU",
                "no_open_set_training":True},temporary)
    temporary.replace(checkpoint)
    metrics = {"history":history,"best_validation_prediction_loss":best,"training_code_prior":prior.tolist(),
               "thresholds":thresholds,"split_episodes":{k:len(v) for k,v in sets.items()},
               "checkpoint_sha256":file_hash(checkpoint),"training_device":str(device),"single_training_seed":True,
               "heldout_online":heldout}
    atomic_json(root/"training_metrics.json",metrics)
    return metrics


def load_qom(path,device):
    payload = torch.load(path,map_location=device,weights_only=False)
    model = QuantizedOpponentModel(payload["codes"],payload["latent"]).to(device)
    model.load_state_dict(payload["model"],strict=True)
    model.eval()
    def factory():
        return QOMRuntime(model,device,payload["prior"],payload["thresholds"])
    return model,factory
