import gym
import numpy as np
import random
from tqdm import tqdm

def random_execution(env, num_steps=99):
    """ This function executes random movements. Each time that we call 
        action_space.sample() it generates a random movement, which is 
        performed once we execute env.step(action). Finally, by executing
        env.render() we are able to see it by a UI.
    Args:
        env (gym.make): Variable used to simulate and generate new states.
        num_steps (int, optional): Number of steps that will be performed. 
                                    Defaults to 99.
    """
    _ = env.reset()
    for s in tqdm(range(num_steps+1)):
        action = env.action_space.sample()
        env.step(action)
        env.render()
    env.close()

def q_learning_execution(env, learning_rate=0.9, discount_rate=0.8, epsilon=1.0, decay_rate=0.005, num_episodes=100, max_steps=99):
    """_summary_

    Args:
        env (gym.make): Variable used to simulate and generate new states.
        learning_rate (float, optional): Defaults to 0.9.
        discount_rate (float, optional): Defaults to 0.8.
        epsilon (float, optional): Defaults to 1.0.
        decay_rate (float, optional): Defaults to 0.005.
        num_episodes (int, optional): Number of episodes that will be performed. Defaults to 100.
        max_steps (int, optional): Number of steps that will be performed. Defaults to 99.
    """
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        for s in tqdm(range(max_steps)):
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(qtable[state,:])
            new_state, reward, done, _, info = env.step(action)
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])
            state = new_state
            if done == True:
                break
        epsilon = np.exp(-decay_rate*episode)
        
    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    state = env.reset()[0]
    done = False
    rewards = 0
    for s in range(max_steps):
        print(f"TRAINED AGENT")
        print("Step {}".format(s+1))
        action = np.argmax(qtable[state,:])
        new_state, reward, done, _, info = env.step(action)
        rewards += reward
        env.render()
        print(f"score: {rewards}")
        state = new_state
        if done == True:
            break
    env.close()

env = gym.make('Taxi-v3', render_mode="human")
q_learning_execution(env)

