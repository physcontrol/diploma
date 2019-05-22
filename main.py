import numpy as np
import scipy as sp
import gym
import random
import time
import os
import matplotlib
import matplotlib.pyplot as plt

env = gym.make("Taxi-v2")
env.render()

action_size = env.action_space.n
state_size = env.observation_space.n

qtable = np.zeros((state_size, action_size))

total_episodes = 50000        # Total episodes
total_test_episodes = 200     # Total test episodes
max_steps = 99                # Max steps per episode

#0 < ... <= 1
alpha = 0.63                   # Learning rate
gamma = 0.75                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate

max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01

score_ov_time =[]
x_steps = []
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    
    for step in range(max_steps):
        ##randomize a number
        exp_exp_tradeoff = random.uniform(0,1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + alpha [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + alpha * (reward + gamma * 
                                    np.max(qtable[new_state, :]) - qtable[state, action])
                
        #new state is state
        state = new_state
        
        # If done : finish episode
        if done == True: 
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

env.reset()
rewards = []

def plotting(action, step, done, info):
    print("ACTION: ", action)
    print("STEP: ", step)
    print("DONE: ", done)
    print("INFO: ", info)
    print("TOTAL REWARD: ", total_rewards)

for episode in range(total_test_episodes):
    state = env.reset()
    x_steps.append(episode / total_test_episodes)
    step = 0
    done = False
    total_rewards = 0
    #os.system("clc||clear")
    #print("****************************************************")
    #print("EPISODE ", episode)

    for step in range(max_steps):
        # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
        # env.render()
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info = env.step(action)
        total_rewards += reward
        #plotting(action, step, done, info)
        if done:
            rewards.append(total_rewards)
            #print ("Score", total_rewards)
            #score_ov_time.append(total_rewards)
            #time.sleep(1)
            break
        state = new_state
    score_ov_time.append(total_rewards)
env.close()
#print(qtable)
print ("Score over time: " +  str(sum(rewards)/total_test_episodes))
# For graphics
#print(len(x_steps), len(score_ov_time))
poly = sp.polyfit(x_steps, score_ov_time, 1)
pol_1d = sp.poly1d(poly)
line_1, line_2 = plt.plot(x_steps, score_ov_time, 'b:', x_steps, pol_1d(x_steps), 'r--')
plt.legend((line_1, line_2), (u'Result', u'Linear Approximation'), loc='upper left')
#plt.title('Зависимость суммарного вознаграждения от пройденного времени\n при N='+str(total_episodes) + ', gamma=' + str(gamma)+', alpha='+str(alpha)+', epsilon='+str(round(epsilon,2)), fontsize=10)
plt.xlabel('Время', fontsize=10)
plt.ylabel('Суммарное вознаграждение', fontsize=10)
graphic_name = 'new_result_1'
#graphic_name = 'withoutlayreward_N' + str(total_episodes)+ 'gamma' + str(gamma)+ 'alpha'+str(alpha)+'epsilon'+str(epsilon)
plt.grid()
plt.savefig(graphic_name)
plt.show()
