import env_factory
from ddpg import *
import gc
import gym
import tensorflow as tf
gc.enable()

ENV_NAME = '3linkarm-v0'
EPISODES = 100000
TEST = 10

def main():
    env_factory.register_env(ENV_NAME)
    env = gym.envs.make(ENV_NAME)
    agent = DDPG(env)
    #env.monitor.start('experiments/' + ENV_NAME,force=True)

    for episode in xrange(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in xrange(env.spec.timestep_limit):
            action = agent.noise_action(state,env.env.goal_state)
            #print agent.actor_network.parameters_gradients
            env.env.actor_network = agent.actor_network
            #print env.actor_network.target_update
            next_state,reward,done,_ = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 100 == 0 and episode > 100:
			total_reward = 0
			for i in xrange(TEST):
				state = env.reset()
				for j in xrange(env.spec.timestep_limit):
					#env.render()
					action = agent.action(state) # direct action for test
					state,reward,done,_ = env.step(action)
					total_reward += reward
					if done:
						break
			ave_reward = total_reward/TEST
			print 'episode: ',episode,'Evaluation Average Reward:',ave_reward
    #file_writer = tf.summary.FileWriter('/path/to/logs', sess.graph)
    # env.monitor.close()

if __name__ == '__main__':
    main()
