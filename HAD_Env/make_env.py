from HAD_Env.env.env import Env


class HADEnv(Env):
    def __init__(self, red_attack_n, blue_attack_n, target_n,
                 red_scout_n=0, red_disturb_n=0, blue_scout_n=0, blue_disturb_n=0,
                 task_type='Training'):
        super(HADEnv, self).__init__(red_scout_n, red_disturb_n, red_attack_n,
                                  blue_scout_n, blue_disturb_n, blue_attack_n,
                                  target_n)

        self.task_type = task_type

    def reset(self, evaluate=False):
        if self.task_type == 'Normal Showcase':
            self.env_uniform_reset()
        else:
            self.env_random_reset(evaluate=evaluate)

        observation_n = self.get_observation(is_relative_observation=True)
        world_alive = self.get_world_alive()
        global_state = self.get_global_state()
        return observation_n, world_alive, global_state

    def step(self, action_n):
        super().step(action_n)

        obs_n = self.get_observation(is_relative_observation=True)
        world_alive_n = self.get_world_alive()
        global_state = self.get_global_state()
        reward_n = self.get_reward()
        done_n = self.get_done()
        info = self.get_info()

        return obs_n, world_alive_n, global_state, reward_n, done_n, info


