"""Entity-input migration of the official SPECTra GRF controller.
Source: funny-rl/SPECTra ffababf6187216c9d16b2109ee8ef6fe5fdf1172.
Football obs splitting is replaced by ALMA's entity contract.
"""
from controllers.entity_controller import EntityMAC
from open_score.models.entity_encoder import PolicyValueMixin
from .spectra_rnn_agent import SPECTra_RNNAgent


class SPECTraMAC(PolicyValueMixin, EntityMAC):
    def _build_agents(self, input_shapes):
        self.agent = SPECTra_RNNAgent(input_shapes[0], self.args)

    def forward(self, ep_batch, t=None, coach_z=None, acting=False,
                test_mode=False, target=False, imagine_inps=None):
        single = isinstance(t, int)
        selected = slice(t, t + 1) if single else (t or slice(0, ep_batch.max_seq_length))
        inputs, _ = self._build_inputs(ep_batch, selected, target=target)
        q, self.hidden_states = self.agent(inputs, self.hidden_states)
        if single:
            self._decision_q = q[:, 0].detach()
        return (q[:, 0] if single else q), {}

    def init_hidden(self, batch_size, n_agents=None):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def set_evaluation_mode(self):
        self.eval()

    def set_train_mode(self):
        self.train()

    def get_device(self):
        return next(self.agent.parameters()).device

