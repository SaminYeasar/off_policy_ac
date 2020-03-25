import gym
import time

env = gym.make('HalfCheetah-v2')

saved_state = None
for _ in range(10):
	time.sleep(10)
	state = env.reset()
	if saved_state is not None:
		state = env.sim.set_state(saved_state)
	done = False

	for _ in range(500):
		env.render()
		array_ = env.step(env.action_space.sample())
	saved_state = env.sim.get_state()
	env.close()