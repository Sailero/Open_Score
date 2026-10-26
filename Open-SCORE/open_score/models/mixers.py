"""Entity adapters around the official ALMA and SPECTra mixers."""
import torch as th
from torch.utils.checkpoint import checkpoint
from modules.mixers.flex_qmix import FlexQMixer




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
    if args.mixer == "flex_qmix":
        return ChunkedFlexQMixer(args)
    if args.mixer == "spectra_mixer":
        from open_score.algos.spectra_patch.spectra_mixer import SPECTraMixer
        return SPECTraMixer(args)
    raise ValueError(f"Unsupported HAD mixer: {args.mixer}")
