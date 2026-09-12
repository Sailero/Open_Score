"""DCG official Algorithms 1--3 adapted to padded variable-size entity batches.
Source: wendelinboehmer/dcg 4de100cddf7c3a7035cd89a47d7c1b8a878e7428.
Retained: shared utility/payoff MLPs, symmetric payoff tables, full-rank default,
double-precision Max-Sum and anytime joint-value selection. Adaptations:
REFIL entity input, initial-node/edge masks, native scatter_add, exact STAR
conditional traceback, and ALMA's controller/checkpoint interfaces.
"""
from controllers.entity_controller import EntityMAC
from .rnn_feature_agent import RNNFeatureAgent
import torch as th
import torch.nn as nn
import numpy as np
import contextlib
import itertools


class DeepCoordinationGraphMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.n_actions = args.n_actions
        self.payoff_rank = args.cg_payoff_rank
        self.payoff_decomposition = isinstance(self.payoff_rank, int) and self.payoff_rank > 0
        self.iterations = args.msg_iterations
        self.normalized = args.msg_normalized
        self.anytime = args.msg_anytime
        self.utility_fun = self._mlp(args.rnn_hidden_dim, args.cg_utilities_hidden_dim, self.n_actions)
        payoff_out = 2 * self.payoff_rank * self.n_actions if self.payoff_decomposition else self.n_actions ** 2
        self.payoff_fun = self._mlp(2 * args.rnn_hidden_dim, args.cg_payoffs_hidden_dim, payoff_out)
        if args.duelling:
            raise ValueError("This protocol uses official DCG, not the optional DCG-V arm")
        self.duelling = False
        self._set_edges(self._edge_list(args.cg_edges))
        self.node_active = None

    def _build_agents(self, input_shapes):
        self.agent = RNNFeatureAgent(input_shapes[0], self.args)

    def _batch_nodes(self, ep_batch):
        if "initial_agent_mask" in ep_batch.scheme:
            initial = ep_batch["initial_agent_mask"]
            if initial.ndim == 3:
                initial = initial[:, 0]
        else:
            initial = ep_batch["entity_mask"][:, 0, :self.n_agents]
        self.node_active = ~initial.bool()

    def annotations(self, ep_batch, t, compute_grads=False, actions=None):
        with th.no_grad() if not compute_grads else contextlib.nullcontext():
            self._batch_nodes(ep_batch)
            inputs, _ = self._build_inputs(ep_batch, slice(t, t + 1))
            unfilled = ~ep_batch["filled"][:, t].bool().reshape(-1)
            inputs["entity_mask"] = inputs["entity_mask"].bool() | unfilled[:, None, None]
            self.hidden_states = self.agent(inputs, self.hidden_states)[1]
            f_i = self.utilities(self.hidden_states)
            f_ij = self.payoffs(self.hidden_states)
            self.last_utilities = f_i.detach()
            self.last_payoffs = f_ij.detach()
        return f_i, f_ij

    def utilities(self, hidden_states):
        return self.utility_fun(hidden_states)

    def payoffs(self, hidden_states):
        # Official Algorithm 1: evaluate both endpoint orders, transpose the
        # reverse action axes, then average to enforce pair symmetry.
        n = self.n_actions
        inputs = th.stack([th.cat([hidden_states[:, self.edges_from], hidden_states[:, self.edges_to]], dim=-1),
                           th.cat([hidden_states[:, self.edges_to], hidden_states[:, self.edges_from]], dim=-1)], dim=0)
        output = self.payoff_fun(inputs)
        if self.payoff_decomposition:
            dim = list(output.shape[:-1])
            output = output.reshape(int(np.prod(dim)) * self.payoff_rank, 2, n)
            output = th.bmm(output[:, 0].unsqueeze(-1), output[:, 1].unsqueeze(-2))
            output = output.reshape(*(dim + [self.payoff_rank, n, n])).sum(dim=-3)
        else:
            output = output.reshape(*(list(output.shape[:-1]) + [n, n]))
        return (output[0] + output[1].transpose(-2, -1)) * 0.5

    def _masks(self, f_i):
        active = self.node_active
        if active is None or active.shape[0] != f_i.shape[0]:
            active = th.ones(f_i.shape[:2], dtype=th.bool, device=f_i.device)
        active = active.to(f_i.device)
        edges = active[:, self.edges_from] & active[:, self.edges_to]
        return active, edges

    def q_values(self, f_i, f_ij, actions):
        # Official Algorithm 2, replacing padded-upper-bound means by means
        # over INITIAL nodes/edges. Dead real nodes retain action zero.
        active, edges = self._masks(f_i)
        values = (f_i.gather(-1, actions).squeeze(-1) * active).sum(-1) / active.sum(-1).clamp_min(1)
        if len(self.edges_from):
            tables = f_ij.reshape(actions.shape[0], len(self.edges_from), self.n_actions ** 2)
            edge_actions = actions[:, self.edges_from] * self.n_actions + actions[:, self.edges_to]
            values += (tables.gather(-1, edge_actions).squeeze(-1) * edges).sum(-1) / edges.sum(-1).clamp_min(1)
        return values

    def star_greedy(self, f_i, f_ij, available_actions=None):
        active, edges = self._masks(f_i)
        utilities = f_i.double() / active.sum(-1).clamp_min(1)[:, None, None]
        if available_actions is not None:
            utilities = utilities.masked_fill(available_actions == 0, -float("inf"))
        tables = f_ij.double() / edges.sum(-1).clamp_min(1)[:, None, None, None]
        root_scores = utilities[:, 0].clone()
        backs = []
        for edge, leaf in enumerate(self.edges_to.tolist()):
            joint = tables[:, edge] + utilities[:, leaf, None, :]
            leaf_scores, leaf_back = joint.max(-1)
            root_scores += leaf_scores.masked_fill(~edges[:, edge, None], 0)
            backs.append(leaf_back)
        root_action = root_scores.argmax(-1)
        actions = utilities.argmax(-1, keepdim=True)
        actions[:, 0, 0] = root_action
        for leaf, back in zip(self.edges_to.tolist(), backs):
            actions[:, leaf, 0] = back.gather(1, root_action[:, None])[:, 0]
        return actions.masked_fill(~active[..., None], 0)

    def greedy(self, f_i, f_ij, available_actions=None):
        if self.args.cg_edges == "star":
            return self.star_greedy(f_i, f_ij, available_actions)
        active, edges = self._masks(f_i)
        in_f_i, in_f_ij = f_i, f_ij
        f_i = f_i.double() / active.sum(-1).clamp_min(1)[:, None, None]
        f_ij = f_ij.double() / edges.sum(-1).clamp_min(1)[:, None, None, None]
        f_ij = f_ij.masked_fill(~edges[..., None, None], 0)
        if available_actions is None:
            legal = th.ones_like(f_i, dtype=th.bool)
        else:
            legal = available_actions.bool()
        if not legal.any(-1).all():
            raise ValueError("Every DCG node, including padding, needs one legal action")
        f_i = f_i.masked_fill(~legal, -float("inf"))
        utils = f_i
        best_actions = utils.argmax(-1, keepdim=True)
        best_value = f_i.new_full((f_i.shape[0],), -float("inf"))
        if len(self.edges_from) and self.iterations > 0:
            messages = f_i.new_zeros(2, f_i.shape[0], len(self.edges_from), self.n_actions)
            destination_legal = th.stack((legal[:, self.edges_to], legal[:, self.edges_from]))
            valid_message = destination_legal & edges[None, :, :, None]
            for _ in range(self.iterations):
                # Official Algorithm 3: remove the opposite-direction message
                # before maximizing over the sender's legal actions.
                joint0 = (utils[:, self.edges_from] - messages[1]).unsqueeze(-1) + f_ij
                joint1 = (utils[:, self.edges_to] - messages[0]).unsqueeze(-1) + f_ij.transpose(-2, -1)
                messages = th.stack((joint0.max(-2).values, joint1.max(-2).values))
                messages = messages.masked_fill(~valid_message, 0)
                if self.normalized:
                    mean = messages.sum(-1, keepdim=True) / valid_message.sum(-1, keepdim=True).clamp_min(1)
                    messages = (messages - mean).masked_fill(~valid_message, 0)
                msg = th.zeros_like(f_i)
                msg.scatter_add_(1, self.edges_to[None, :, None].expand(f_i.shape[0], -1, self.n_actions), messages[0])
                msg.scatter_add_(1, self.edges_from[None, :, None].expand(f_i.shape[0], -1, self.n_actions), messages[1])
                utils = f_i + msg
                if self.anytime:
                    actions = utils.argmax(-1, keepdim=True)
                    value = self.q_values(in_f_i, in_f_ij, actions)
                    change = value > best_value
                    best_value = th.where(change, value, best_value)
                    best_actions = th.where(change[:, None, None], actions, best_actions)
        if not self.anytime:
            best_actions = utils.argmax(-1, keepdim=True)
        return best_actions.masked_fill(~active[..., None], 0)

    def forward(self, ep_batch, t, actions=None, policy_mode=True,
                test_mode=False, compute_grads=False, **kwargs):
        f_i, f_ij = self.annotations(ep_batch, t, compute_grads, actions)
        if actions is not None and not policy_mode:
            return self.q_values(f_i, f_ij, actions)
        available = ep_batch["avail_actions"][:, t].clone()
        # Padding after an already completed episode has no real transition;
        # give its dummy rows a no-op without concealing invalid real states.
        unfilled = ~ep_batch["filled"][:, t].bool().reshape(-1)
        available[unfilled, :, 0] = 1
        actions = self.greedy(f_i, f_ij, available)
        if policy_mode:
            policy = th.zeros_like(f_i).scatter_(-1, actions, 1)
            return policy, {"utilities": f_i}
        return actions

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        policy, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            policy[bs], ep_batch["avail_actions"][bs, t_ep], t_env, test_mode=test_mode)

    def cuda(self):
        self.agent.cuda()
        self.utility_fun.cuda()
        self.payoff_fun.cuda()
        self.edges_from = self.edges_from.cuda()
        self.edges_to = self.edges_to.cuda()

    def parameters(self):
        return itertools.chain(self.agent.parameters(), self.utility_fun.parameters(), self.payoff_fun.parameters())

    def train(self):
        self.agent.train()
        self.utility_fun.train()
        self.payoff_fun.train()

    def eval(self):
        self.agent.eval()
        self.utility_fun.eval()
        self.payoff_fun.eval()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())
        self.utility_fun.load_state_dict(other_mac.utility_fun.state_dict())
        self.payoff_fun.load_state_dict(other_mac.payoff_fun.state_dict())

    def save_models(self, path):
        th.save(self.agent.state_dict(), f"{path}agent.th")
        th.save(self.utility_fun.state_dict(), f"{path}utilities.th")
        th.save(self.payoff_fun.state_dict(), f"{path}payoffs.th")

    def state_dict(self):
        return {"agent": self.agent.state_dict(), "utilities": self.utility_fun.state_dict(),
                "payoffs": self.payoff_fun.state_dict()}

    def load_state_dict(self, state):
        self.agent.load_state_dict(state["agent"])
        self.utility_fun.load_state_dict(state["utilities"])
        self.payoff_fun.load_state_dict(state["payoffs"])

    def evaluation_values(self, batch, t, actions, active, mixer=None):
        active = list(active)
        fi, fij = self.last_utilities[active], self.last_payoffs[active]
        actions = th.as_tensor(actions, device=fi.device, dtype=th.long)
        if actions.ndim == 2:
            actions = actions.unsqueeze(-1)
        initial_active = self.node_active
        self.node_active = initial_active[active]
        try:
            total = self.q_values(fi, fij, actions)
        finally:
            self.node_active = initial_active
        alive = ~batch["entity_mask"][active, t, :self.n_agents].bool()
        qi = fi.gather(-1, actions).squeeze(-1).masked_fill(~alive, 0)
        qi = qi.sum(-1) / alive.sum(-1).clamp_min(1)
        return {"q_tot": total.detach().cpu().tolist(), "q_i": qi.detach().cpu().tolist()}

    def load_models(self, path, pi_only=False):
        self.agent.load_state_dict(th.load(f"{path}agent.th", map_location="cpu"))
        self.utility_fun.load_state_dict(th.load(f"{path}utilities.th", map_location="cpu"))
        self.payoff_fun.load_state_dict(th.load(f"{path}payoffs.th", map_location="cpu"))

    @staticmethod
    def _mlp(input, hidden_dims, output):
        hidden_dims = [] if hidden_dims is None else hidden_dims
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        dim, layers = input, []
        for d in hidden_dims:
            layers.extend((nn.Linear(dim, d), nn.ReLU()))
            dim = d
        layers.append(nn.Linear(dim, output))
        return nn.Sequential(*layers)

    def _edge_list(self, arg):
        if arg == "full":
            return [(i, j) for i in range(self.n_agents) for j in range(i + 1, self.n_agents)]
        if arg == "star":
            return [(0, j) for j in range(1, self.n_agents)]
        if arg == "vdn":
            return []
        if isinstance(arg, list):
            return arg
        raise ValueError("This protocol supports full, star, vdn or explicit edge pairs")

    def _set_edges(self, edge_list):
        self.edges_from = th.tensor([e[0] for e in edge_list], dtype=th.long)
        self.edges_to = th.tensor([e[1] for e in edge_list], dtype=th.long)

