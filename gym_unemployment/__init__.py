from gym.envs.registration import register

register(
    id='unemployment-v0',
    entry_point='gym_unemployment.envs:UnemploymentEnv_v0',
)

register(
    id='unemployment-rev-v0',
    entry_point='gym_unemployment.envs:UnemploymentRevEnv_v0',
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
    id='unemployment-v4',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v4',
)

register(
    id='unemployment-v5',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v5',
)

register(
    id='unemployment-v6',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v6',
)

register(
    id='unemployment-v7',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v7',
)

register(
    id='unemployment-v8',
    entry_point='gym_unemployment.envs:UnemploymentLargeEnv_v8',
)

register(
    id='unemployment-v9',
    entry_point='gym_unemployment.envs:UnemploymentEnv_v9',
)

register(
    id='Qunemployment-v7',
    entry_point='gym_unemployment.envs:QUnemploymentLargeEnv_v7',
)

#register(
#    id='unemploymentEK-v0',
#    entry_point='gym_unemployment.envs:UnemploymentEKEnv',
#)

#register(
#    id='unemploymentEK-v1',
#    entry_point='gym_unemployment.envs:UnemploymentEKLargeEnv',
#)

register(
    id='unemploymentEK-v3',
    entry_point='gym_unemployment.envs:UnemploymentEKLargeEnv_v3',
)

register(
    id='unemploymentEK-v4',
    entry_point='gym_unemployment.envs:UnemploymentEKEnv_v4',
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

register(
    id='savings-v0',
    entry_point='gym_unemployment.envs:SavingsEnv_v0',
)


register(
    id='megasavings-v0',
    entry_point='gym_unemployment.envs:MegaSavingsEnv_v0',
)

