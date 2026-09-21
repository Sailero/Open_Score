"""Entity adapters around the official ALMA and SPECTra mixers."""
from copy import copy
import torch as th
from torch.utils.checkpoint import checkpoint
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer


class PaddedQMixer(QMixer):
    """B0 keeps the ordinary order-sensitive QMIX hypernetwork.

    ``pool_slots`` sizes both the flattened state and the ordered agent mixing
    rows to the training-pool roster, leaving no untrained state column or
    mixing row behind. A roster beyond that budget is refused where the
    configuration is chosen, not here: unfilled replay timesteps carry an
    all-zero mask and would otherwise look like a full pad roster.
    """
    def __init__(self, args):
        from open_score.envs.features import pool_slot_indices
        keep = pool_slot_indices(args)
        adapted = copy(args)
        dim = args.entity_shape + (args.n_actions if args.entity_last_action else 0)
        n_slots = args.n_entities if keep is None else len(keep)
        if keep is not None:
            adapted.n_agents = int(args.pool_slots[0])
        adapted.state_shape = n_slots * (dim + 1)
        super().__init__(adapted)
        self.register_buffer("keep", None if keep is None else
                             th.as_tensor(keep, dtype=th.long), persistent=False)

    def forward(self, agent_qs, inputs, imagine_groups=None):
        alive = ~inputs["entity_mask"].bool()
        entities = inputs["entities"].masked_fill(~alive.unsqueeze(-1), 0)
        state = th.cat((entities, alive.to(entities.dtype).unsqueeze(-1)), dim=-1)
        agent_qs = agent_qs.masked_fill(~alive[..., :agent_qs.shape[-1]], 0)
        if self.keep is not None:
            state = state.index_select(2, self.keep)
            agent_qs = agent_qs[..., :self.n_agents]
        return super().forward(agent_qs, state.flatten(2), imagine_groups=imagine_groups)

    def denormalize(self, q):
        return q


class ChunkedFlexQMixer(FlexQMixer):
    """Run the official mixer on independent state blocks.

    The inherited ALMA hypernetworks, nonnegative weights and REFIL group
    masks are unchanged. Only the batch/time execution layout and saved
    activations differ; state_dict keys stay compatible with existing runs.
    A nonpositive mixer_chunk_size selects the original unchunked reference.
    """
    def forward(self, agent_qs, inputs, imagine_groups=None):
        chunk_size = int(getattr(self.args, "mixer_chunk_size", 256))
        if chunk_size <= 0:
            return super().forward(agent_qs, inputs, imagine_groups=imagine_groups)
        bs, ts = inputs["entities"].shape[:2]
        # Official FlexQMixer already flattens B*T. Flattening before the
        # split bounds every call even when batch_size exceeds chunk_size.
        flat_qs = agent_qs.reshape(bs * ts, -1)
        flat_inputs = {key: value.reshape(bs * ts, *value.shape[2:])
                       for key, value in inputs.items() if key != "state_mask"}
        flat_groups = None if imagine_groups is None else tuple(
            group.reshape(bs * ts, *group.shape[2:]) for group in imagine_groups)
        active = None
        if "state_mask" in inputs and getattr(self.args, "mixer_skip_unfilled", True):
            # Only replay time-padding is dispensable. A real final state
            # (including all-red-dead or truncation) remains if filled=1.
            active = inputs["state_mask"].reshape(bs * ts).bool().nonzero(as_tuple=False).flatten()
            flat_qs = flat_qs[active]
            flat_inputs = {key: value[active] for key, value in flat_inputs.items()}
            if flat_groups is not None:
                flat_groups = tuple(group[active] for group in flat_groups)
        if flat_qs.shape[0] == 0:
            n_out = self.args.n_tasks if self.args.mixer_subtask_cond is not None else 1
            result = flat_qs.new_zeros(bs, ts, n_out)
            if th.is_grad_enabled():
                zero = flat_qs.sum() * 0
                for parameter in self.parameters():
                    zero = zero + parameter.reshape(-1)[0] * 0
                result = result + zero
            return result
        outputs = []
        for start in range(0, flat_qs.shape[0], chunk_size):
            stop = start + chunk_size
            qs = flat_qs[start:stop].unsqueeze(0)
            part = {key: value[start:stop].unsqueeze(0) for key, value in flat_inputs.items()}
            groups = None if flat_groups is None else tuple(
                group[start:stop].unsqueeze(0) for group in flat_groups)
            if self.training and th.is_grad_enabled():
                total = checkpoint(super().forward, qs, part, imagine_groups=groups,
                                   use_reentrant=False)
            else:
                total = super().forward(qs, part, imagine_groups=groups)
            outputs.append(total)
        totals = th.cat(outputs, dim=1).squeeze(0)
        if active is not None:
            totals = totals.new_zeros(bs * ts, totals.shape[-1]).index_copy(0, active, totals)
        return totals.reshape(bs, ts, -1)


def build_mixer(args):
    if args.mixer == "transfqmix":
        from open_score.algos.transfqmix import TransfQMixMixer
        return TransfQMixMixer(args)
    if args.mixer in (None, "none"):
        return None
    if args.mixer == "qmix":
        return PaddedQMixer(args)
    if args.mixer == "flex_qmix":
        return ChunkedFlexQMixer(args)
    if args.mixer == "spectra_mixer":
        from open_score.algos.spectra_patch.spectra_mixer import SPECTraMixer
        return SPECTraMixer(args)
    raise ValueError(f"Unsupported HAD mixer: {args.mixer}")
