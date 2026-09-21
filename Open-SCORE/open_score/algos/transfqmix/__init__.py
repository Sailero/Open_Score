"""TransfQMix (Gallici et al., AAMAS 2023), adapted to EpisodeBatch.

Independent implementation of the paper's recurrent actor and hypernetwork.
Operation reference: https://github.com/mttga/pymarl_transformers at
2ef0a0726f1f186097b4b560509ceb5a801fdafe, modules/layer/transformer.py,
modules/agents/n_transf_agent{,_smac}.py, modules/mixers/n_transf_mixer.py,
and learners/nq_transf_learner.py. No upstream license is asserted here.
The entity masks and three-token mixer initialization are explicit adaptations.
"""
import copy

import torch as th
from torch import nn
from torch.nn import functional as F


class _Attention(nn.Module):
    """Each head has the full embedding width, as in the author code."""

    def __init__(self, width, heads):
        super().__init__()
        self.width, self.heads = width, heads
        self.keys = nn.Linear(width, width * heads, bias=False)
        self.queries = nn.Linear(width, width * heads, bias=False)
        self.values = nn.Linear(width, width * heads, bias=False)
        self.output = nn.Linear(width * heads, width)

    def forward(self, query, source, excluded):
        batch, count, width = source.shape
        def split(tensor):
            return tensor.reshape(batch, -1, self.heads, width).transpose(1, 2)
        k, q, v = split(self.keys(source)), split(self.queries(query)), split(self.values(source))
        scores = (q / width ** .25) @ (k / width ** .25).transpose(-1, -2)
        scores = scores.masked_fill(excluded[:, None, None, :], -th.inf)
        weights = scores.softmax(dim=-1)
        combined = (weights @ v).transpose(1, 2).reshape(batch, query.shape[1], -1)
        return self.output(combined)


class _TransformerBlock(nn.Module):
    def __init__(self, width, heads, ff_mult, dropout):
        super().__init__()
        self.attention = _Attention(width, heads)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_mult * width), nn.ReLU(),
                                nn.Linear(ff_mult * width, width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, source, excluded):
        x = self.dropout(self.norm1(query + self.attention(query, source, excluded)))
        return self.dropout(self.norm2(x + self.ff(x)))


class _Transformer(nn.Module):
    def __init__(self, width, heads, depth, ff_mult, dropout):
        super().__init__()
        self.layers = nn.ModuleList(_TransformerBlock(width, heads, ff_mult, dropout)
                                    for _ in range(depth))

    def forward(self, tokens, excluded):
        # K/V keep the original embedded tokens through every layer. Only Q
        # advances; nn.TransformerEncoder would implement a different model.
        source = tokens
        for layer in self.layers:
            tokens = layer(tokens, source, excluded)
        return tokens


class TransfQMixAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.width = int(getattr(args, "emb", 32))
        self.n_agents = int(args.n_agents)
        self.smac = str(getattr(args, "env", "had")).startswith("sc2") or str(getattr(args, "env", "had")).startswith("smac")
        self.n_enemies = int(getattr(args, "n_enemies", 0)) if self.smac else 0
        if self.smac and (self.n_enemies < 1 or args.n_actions != 6 + self.n_enemies):
            raise ValueError("TransfQMix SMAC requires 6 basic actions plus one action per enemy")
        self.feat_embedding = nn.Linear(input_shape, self.width)
        self.transformer = _Transformer(self.width, int(getattr(args, "heads", 4)),
            int(getattr(args, "depth", 2)), int(getattr(args, "ff_hidden_mult", 4)),
            float(getattr(args, "dropout", 0)))
        self.q_basic = nn.Linear(self.width, 6 if self.smac else args.n_actions)
        if self.smac:
            self.q_entity = nn.Linear(self.width, 1)

    def init_hidden(self):
        return self.feat_embedding.weight.new_zeros(1, self.width)

    def forward(self, entities, excluded, hidden, dead):
        # The recurrent token always remains a valid key, even for dead agents.
        entities = entities.masked_fill(excluded[..., None], 0)
        embedded = self.feat_embedding(entities).masked_fill(excluded[..., None], 0)
        hidden = hidden.reshape(-1, 1, self.width).masked_fill(dead[:, None, None], 0)
        tokens = th.cat((hidden, embedded), dim=1)
        mask = th.cat((th.zeros_like(excluded[:, :1]), excluded), dim=1)
        output = self.transformer(tokens, mask)
        recurrent = output[:, 0].masked_fill(dead[:, None], 0)
        q = self.q_basic(recurrent)
        if self.smac:
            # MAC explicitly maps friend-first native slots to enemy-first;
            # output attack i consequently addresses native enemy i.
            attacks = self.q_entity(output[:, 1:1 + self.n_enemies]).squeeze(-1)
            q = th.cat((q, attacks), dim=-1)
        return q.masked_fill(dead[:, None], 0), recurrent


class TransfQMixMAC:
    """Existing ParallelRunner controller contract with cached recurrent output."""

    def __init__(self, scheme, groups, args):
        from components.action_selectors import REGISTRY
        self.args, self.n_agents = args, int(args.n_agents)
        self.use_alloc = self.learned_alloc = self.use_copa = False
        self.agent_output_type = "q"
        shape = scheme.get("observer_entities", scheme["entities"])["vshape"]
        width = int(shape[-1] if isinstance(shape, (tuple, list)) else shape)
        self.agent = TransfQMixAgent(width, args)
        if self.agent.smac and "observer_entities" not in scheme:
            raise ValueError("TransfQMix SMAC actor requires native observer_entities")
        self.action_selector = REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self._evaluation_hyper = None

    def init_hidden(self, batch_size, n_agents=None):
        if n_agents is not None and n_agents != self.n_agents:
            raise ValueError("Controller agent count must agree with its EpisodeBatch")
        self.hidden_states = self.agent.init_hidden().expand(batch_size, self.n_agents, -1).clone()
        self._evaluation_hyper = None
        self._evaluation_t = None
        return self.hidden_states

    def _step(self, batch, t):
        absent = batch["entity_mask"][:, t].bool()
        if "filled" in batch.scheme:
            absent = absent | ~batch["filled"][:, t].bool()
        dead = absent[:, :self.n_agents]
        if self.agent.smac:
            entities = batch["observer_entities"][:, t]
            mask = batch["obs_mask"][:, t, :self.n_agents].bool() | absent[:, None]
            ne = self.agent.n_enemies
            if entities.shape[2] != self.n_agents + ne:
                raise ValueError("SMAC entity order requires exactly friends then enemies")
            order = th.cat((th.arange(self.n_agents, self.n_agents + ne, device=entities.device),
                            th.arange(self.n_agents, device=entities.device)))
            entities = entities.index_select(2, order).flatten(0, 1)
            mask = mask.index_select(2, order).flatten(0, 1)
        else:
            # Import at call time: the shared model module also imports MACs.
            from open_score.models.entity_encoder import relative_entity_views
            inputs = {"entities": batch["entities"][:, t:t + 1],
                      "obs_mask": batch["obs_mask"][:, t:t + 1],
                      "entity_mask": absent[:, None]}
            entities, mask, _, _ = relative_entity_views(inputs, self.args)
        q, hidden = self.agent(entities, mask, self.hidden_states, dead.flatten())
        self.hidden_states = hidden.reshape(batch.batch_size, self.n_agents, -1)
        return q.reshape(batch.batch_size, self.n_agents, -1), self.hidden_states

    def forward(self, ep_batch, t=None, return_hs=False, **kwargs):
        integer = isinstance(t, int)
        interval = slice(t, t + 1) if integer else (t or slice(0, ep_batch.max_seq_length))
        start, stop, step = interval.indices(ep_batch.max_seq_length)
        if step != 1:
            raise ValueError("Recurrent controller requires consecutive timesteps")
        outputs, states = [], []
        for current in range(start, stop):
            q, hidden = self._step(ep_batch, current)
            outputs.append(q)
            states.append(hidden)
        q, hidden = th.stack(outputs, 1), th.stack(states, 1)
        if integer:
            q, hidden = q[:, 0], hidden[:, 0]
            self._decision_q, self._decision_hidden = q.detach(), hidden.detach()
        return q, hidden if return_hs else {"hidden_states": hidden}

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        q, _ = self.forward(ep_batch, t_ep)
        return self.action_selector.select_action(q[bs], ep_batch["avail_actions"][bs, t_ep],
                                                  t_env, test_mode=test_mode)

    def evaluation_values(self, batch, t, actions, active, mixer):
        active = list(active)
        q = self._decision_q[active]
        actions = th.as_tensor(actions, device=q.device, dtype=th.long).reshape(q.shape[:2])
        chosen = q.gather(-1, actions[..., None]).squeeze(-1)
        alive = ~batch["entity_mask"][active, t, :self.n_agents].bool()
        chosen = chosen.masked_fill(~alive, 0)
        individual = chosen.sum(-1) / alive.sum(-1).clamp_min(1)
        total = individual
        if mixer is not None:
            if self._evaluation_hyper is None:
                self._evaluation_hyper = mixer.init_hidden(batch.batch_size)
            if self._evaluation_t == t:
                total = self._evaluation_total[active]
            else:
                with th.no_grad():
                    total, recurrent = mixer(chosen, self._decision_hidden[active],
                        self._evaluation_hyper[active], batch["entities"][active, t],
                        batch["entity_mask"][active, t])
                self._evaluation_hyper[active] = recurrent
                self._evaluation_total = q.new_zeros(batch.batch_size)
                self._evaluation_total[active] = total.flatten()
                self._evaluation_t = t
        return {"q_tot": total.flatten().detach().cpu().tolist(),
                "q_i": individual.detach().cpu().tolist()}

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def eval(self):
        self.agent.eval()

    def train(self):
        self.agent.train()

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}agent.th")

    def load_models(self, path, pi_only=False):
        self.agent.load_state_dict(th.load(f"{path}agent.th", map_location="cpu"))


class TransfQMixMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_agents = int(args.n_agents)
        self.width = int(getattr(args, "mixer_emb", 32))
        if self.width != int(getattr(args, "emb", 32)):
            raise ValueError("TransfQMix actor and mixer token widths must agree")
        self.feat_embedding = nn.Linear(int(args.entity_shape), self.width)
        self.transformer = _Transformer(self.width, int(getattr(args, "mixer_heads", 4)),
            int(getattr(args, "mixer_depth", 2)), int(getattr(args, "ff_hidden_mult", 4)),
            float(getattr(args, "dropout", 0)))
        self.hyper_b2 = nn.Linear(self.width, 1)

    def init_hidden(self, batch_size=1):
        # Paper: three temporal tokens for b1, w2, b2, independent of team size.
        return self.feat_embedding.weight.new_zeros(batch_size, 3, self.width)

    def forward(self, qvals, agent_hidden, hyper_hidden, entities, entity_mask):
        batch = entities.shape[0]
        absent = entity_mask.bool()
        dead = absent[:, :self.n_agents]
        central = self.feat_embedding(entities.masked_fill(absent[..., None], 0))
        central = central.masked_fill(absent[..., None], 0)
        # Author learner explicitly detaches these tokens: actor learns through
        # chosen Q-values only, while its own temporal recurrence retains BPTT.
        agent_hidden = agent_hidden.detach().masked_fill(dead[..., None], 0)
        tokens = th.cat((central, agent_hidden, hyper_hidden), dim=1)
        excluded = th.cat((absent, dead, th.zeros(batch, 3, dtype=th.bool, device=entities.device)), 1)
        output = self.transformer(tokens, excluded)
        weights = output[:, -3-self.n_agents:-3].abs().masked_fill(dead[..., None], 0)
        bias1 = output[:, -3:-2]
        weights2 = output[:, -2].abs().unsqueeze(-1)
        bias2 = F.relu(self.hyper_b2(output[:, -1])).unsqueeze(-1)
        qvals = qvals.reshape(batch, 1, self.n_agents).masked_fill(dead[:, None], 0)
        mixed = F.elu(qvals @ weights + bias1) @ weights2 + bias2
        return mixed.reshape(batch, 1), output[:, -3:]

    @staticmethod
    def denormalize(values):
        return values


def build_td_lambda_targets(rewards, terminated, mask, target_qs, gamma, td_lambda):
    """TD(lambda), bootstrapping at each padded episode's time-limit boundary."""
    returns = th.zeros_like(target_qs)
    returns[:, -1] = target_qs[:, -1]
    for t in range(rewards.shape[1] - 1, -1, -1):
        next_valid = mask[:, t + 1] if t + 1 < mask.shape[1] else th.zeros_like(mask[:, t])
        continuation = ((1 - td_lambda * next_valid) * target_qs[:, t + 1]
                        + td_lambda * next_valid * returns[:, t + 1])
        returns[:, t] = mask[:, t] * (rewards[:, t] + gamma * (1 - terminated[:, t]) * continuation)
    return returns[:, :-1]


class TransfQMixLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args, self.mac, self.logger = args, mac, logger
        self.mixer = TransfQMixMixer(args)
        self.target_mac, self.target_mixer = copy.deepcopy(mac), copy.deepcopy(self.mixer)
        for parameter in self.target_mac.parameters():
            parameter.requires_grad_(False)
        self.target_mixer.requires_grad_(False)
        self.params = list(mac.parameters()) + list(self.mixer.parameters())
        self.optimiser = th.optim.Adam(self.params, lr=args.lr, weight_decay=args.weight_decay)
        self.last_target_update_episode = 0
        self.log_stats_t = -args.learner_log_interval - 1

    def _rollout(self, mac, batch):
        mac.init_hidden(batch.batch_size)
        return mac.forward(batch, return_hs=True)

    def _mix_sequence(self, mixer, values, hidden, batch):
        recurrent = mixer.init_hidden(batch.batch_size)
        outputs = []
        for t in range(values.shape[1]):
            absent = batch["entity_mask"][:, t].bool() | ~batch["filled"][:, t].bool()
            total, recurrent = mixer(values[:, t], hidden[:, t], recurrent,
                                     batch["entities"][:, t], absent)
            outputs.append(total)
        return th.stack(outputs, dim=1)

    def train(self, batch, t_env, episode_num):
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float().clone()
        mask[:, 1:] *= 1 - batch["reset"][:, :-2].float()
        self.mac.train()
        self.mixer.train()
        q, hidden = self._rollout(self.mac, batch)
        chosen = q[:, :-1].gather(-1, batch["actions"][:, :-1]).squeeze(-1)
        with th.no_grad():
            self.target_mac.eval()
            self.target_mixer.eval()
            target_q, target_hidden = self._rollout(self.target_mac, batch)
            greedy = q.detach().masked_fill(batch["avail_actions"] == 0, -th.inf).argmax(-1, keepdim=True)
            target_chosen = target_q.gather(-1, greedy).squeeze(-1)
            target_total = self._mix_sequence(self.target_mixer, target_chosen, target_hidden, batch)
            targets = build_td_lambda_targets(rewards, terminated, mask, target_total,
                                               self.args.gamma, self.args.td_lambda)
        total = self._mix_sequence(self.mixer, chosen, hidden[:, :-1], batch)
        error = total - targets
        count = mask.sum().clamp_min(1)
        loss = (0.5 * error.square() * mask).sum() / count
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        if not th.isfinite(loss) or not th.isfinite(norm):
            raise FloatingPointError("Non-finite TransfQMix loss or gradient")
        self.optimiser.step()
        if episode_num - self.last_target_update_episode >= self.args.target_update_interval:
            self.target_mac.load_state(self.mac)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update_episode = episode_num
        valid = total.detach()[mask.bool()]
        self.last_metrics = {"loss": float(loss.detach()), "grad_norm": float(norm),
            "td_error_abs": float((error.detach().abs() * mask).sum() / count),
            "q_tot_mean": float(valid.mean()), "q_tot_std": float(valid.std(unbiased=False))}
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            for name, value in self.last_metrics.items():
                self.logger.log_stat("loss_td" if name == "loss" else name, value, t_env)
            self.log_stats_t = t_env
        return self.last_metrics

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), f"{path}mixer.th")
        th.save(self.optimiser.state_dict(), f"{path}opt.th")

    def load_models(self, path, pi_only=False, evaluate=False):
        self.mac.load_models(path)
        self.target_mac.load_state(self.mac)
        self.mixer.load_state_dict(th.load(f"{path}mixer.th", map_location="cpu"))
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        if not pi_only and not evaluate:
            self.optimiser.load_state_dict(th.load(f"{path}opt.th", map_location="cpu"))
