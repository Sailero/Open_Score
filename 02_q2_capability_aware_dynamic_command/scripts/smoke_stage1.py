"""Run one HAD transition and one QMIX gradient step on CPU or CUDA."""

import copy
import json
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(WORKSPACE))

from open_score.envs import HADStage1Adapter, tensorize_had_observation
from open_score.stage1 import QMixTransition, VariableScaleQMIX, one_step_qmix_td_loss


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = HADStage1Adapter(3, 4, 2, controlled_side="Red", max_steps=40)
    first = adapter.reset(seed=7)
    observation, state = tensorize_had_observation(first, device)
    model = VariableScaleQMIX(
        entity_dim=adapter.ENTITY_DIM,
        self_dim=adapter.SELF_DIM,
        task_dim=adapter.TASK_DIM,
        state_entity_dim=adapter.STATE_ENTITY_DIM,
        action_dim=adapter.ACTION_DIM,
    ).to(device)
    target = copy.deepcopy(model).eval()
    actions, _ = model.act(observation, epsilon=1.0)
    second, reward, done, info = adapter.step(actions.squeeze(0).cpu().numpy())
    next_observation, next_state = tensorize_had_observation(second, device)
    transition = QMixTransition(
        observation=observation,
        state=state,
        actions=actions,
        reward=torch.tensor([reward], dtype=torch.float32, device=device),
        next_observation=next_observation,
        next_state=next_state,
        done=torch.tensor([done], dtype=torch.float32, device=device),
    )
    loss = one_step_qmix_td_loss(model, target, transition)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    print(
        json.dumps(
            {
                "device": str(device),
                "controlled_agents": int(observation.agent_mask.sum()),
                "entities": int(state.entity_mask.sum()),
                "reward": reward,
                "done": done,
                "terminated": info["terminated"],
                "loss": float(loss.detach().cpu()),
                "grad_norm": float(grad_norm.detach().cpu()),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
