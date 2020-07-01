
from gym.envs.registration import register

register(
	id='MinitaurTorqueEnv_fastAdapt-v0',
	entry_point='fast_adaptation_embedding.env.minitaur_torque_env:MinitaurTorqueEnv',
	max_episode_steps=10000,
	reward_threshold=10000.0
)

register(
	id='SpotMicroEnv-v0',
	entry_point='fast_adaptation_embedding.env.spotmicro_env:SpotMicroEnv',
	max_episode_steps=10000,
	reward_threshold=10000.0
)

register(
	id='SpotMicroStandupEnv-v0',
	entry_point='fast_adaptation_embedding.env.spotmicro_standup_env:SpotMicroStandupEnv',
	max_episode_steps=10000,
	reward_threshold=10000.0
)

register(
	id='ANYmalEnv-v0',
	entry_point='fast_adaptation_embedding.env.anymal_env:ANYmalEnv',
	max_episode_steps=10000,
	reward_threshold=10000.0
)