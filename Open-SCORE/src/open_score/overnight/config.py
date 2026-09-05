"""Explicit portfolio protocols; these are starting configurations, not optima."""
ROUTES = ('ppo_structured', 'ppo_teacher', 'candidate_q')


def configuration(route, *, seed=20260906, smoke=False, device='cpu'):
    if route not in ROUTES:
        raise ValueError(f'Unknown route: {route}')
    return dict(schema='known-grouping-overnight-v3', route=route, seed=int(seed),
                device=device, executor_device='cpu', executor_scope='count_clip3',
                opponent='reactive', max_steps=50, command_interval=5,
                model=dict(hidden_dim=64, heads=4, layers=2) if smoke else dict(hidden_dim=256, heads=8, layers=3),
                candidate_limit=8 if smoke else 24, environments=2 if smoke else 8,
                rollout_events=32 if smoke else 1024, batch_size=16 if smoke else 128,
                epochs=2 if smoke else 4, learning_rate=2e-4, gamma=1.0,
                gae_lambda=.95, clip=.2, value_coef=.5, target_kl=.03,
                entropy_start=.02, entropy_end=.005, max_gradient_norm=.5,
                replay_capacity=256 if smoke else 12000, replay_warmup=32 if smoke else 512,
                q_target_every=100, q_updates_per_vector_step=1, epsilon_end=.08,
                teacher_fraction=.12, teacher_candidates=3 if smoke else 6,
                teacher_horizon=5 if smoke else 15, teacher_capacity=32 if smoke else 4096,
                teacher_aux_coef=.1, train_scales=[4, 8] if smoke else [4, 8, 12, 16, 24, 32],
                eval_scales=[4, 8] if smoke else [8, 12, 16, 24, 32],
                validation_scales=[4, 8] if smoke else [8, 16, 32],
                validation_episodes=2 if smoke else 12,
                validation_interval=15 if smoke else 1800,
                checkpoint_seconds=10 if smoke else 300,
                smoke=bool(smoke), evaluation_decoding='greedy',
                reward='native_success_plus_fixed_potential_difference_terminal_phi_zero')


def curriculum_scales(config, fraction):
    if config['smoke']:
        return config['train_scales']
    if fraction < .15:
        return [4, 8, 12]
    if fraction < .4:
        return [8, 12, 16, 24]
    return [8, 12, 16, 24, 32]
