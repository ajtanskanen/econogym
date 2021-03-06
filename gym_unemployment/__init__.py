from gym.envs.registration import register

register(
    id='unemployment-v0',
    entry_point='gym_unemployment.envs:UnemploymentEnv_v0',
)

register(
    id='unemployment-v1',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv',
)

register(
    id='unemployment-v2',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v2',
)

register(
    id='unemployment-v3',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v3',
)

register(
    id='unemploymentEK-v0',
    entry_point='gym_unemployment.envs:UnemploymentEKEnv',
)

register(
    id='unemploymentEK-v1',
    entry_point='gym_unemployment.envs:UnemploymentEKLargeEnv',
)

register(
    id='unemploymentSteps-v1',
    entry_point='gym_unemployment.envs:UnemploymentStepsLargeEnv',
)

register(
    id='test-v1',
    entry_point='gym_unemployment.envs:TestEnv',
)

register(
    id='unemployment-long-v0',
    entry_point='gym_unemployment.envs:UnemploymentLongEnv_v0',
)

