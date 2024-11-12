import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import logging
from env_RL_energy_efficient import RaceTrack
import os
import random
import time
import tensorflow as tf
from tensorflow.keras import layers, models

matplotlib.use('TkAgg')

# Configuration du logging
logging.basicConfig(level=logging.INFO)

#Configuration de l'environement
env = RaceTrack(track_map='a', render_mode=None)

class DQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(64, activation='relu')
        self.dense3 = layers.Dense(action_space, activation=None)  # One Q-value for each action

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.dense3(x)


class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration factor (epsilon-greedy)
epsilon_min = 0.01  # Minimum value for epsilon
epsilon_decay = 0.995  # Epsilon decay per episode
learning_rate = 0.001
batch_size = 32

# Create a target Q-network for stable training (we'll update it periodically)
target_dqn = DQN(action_space=9)
optimizer = tf.keras.optimizers.Adam(learning_rate)

def compute_loss(batch, model, target_model):
    states, actions, rewards, next_states, dones = zip(*batch)
    states = np.array(states)
    next_states = np.array(next_states)

    # Calculate the Q-values for current states using the model
    q_values = model(states)
    target_q_values = target_model(next_states)

    # Extract the Q-value of the chosen action
    action_indices = np.array(actions)
    q_values = tf.gather(q_values, action_indices, axis=1, batch_dims=1)

    # Compute the target Q-values
    max_next_q_values = tf.reduce_max(target_q_values, axis=1)
    target = rewards + (gamma * max_next_q_values * (1 - dones))

    # Compute loss using mean squared error (MSE)
    loss = tf.reduce_mean(tf.square(target - q_values))
    return loss

# Train function
def train_step(batch, model, target_model):
    with tf.GradientTape() as tape:
        loss = compute_loss(batch, model, target_model)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Soft behavior policy
def behavior_pi(state: tuple, nA: int, target_pi: np.ndarray, epsilon: float) -> tuple:
    '''
    The behavior policy returns both the action and the probability of that action
    '''
    rand_val = np.random.rand()
    greedy_act = target_pi[state]
    
    if rand_val > epsilon:
        return greedy_act, (1 - epsilon + epsilon / nA)
    else:
        action = np.random.choice(nA)
        if action == greedy_act:
            return action, (1 - epsilon + epsilon / nA)
        else:
            return action, epsilon / nA


# Plot the result
def plot_result(value_hist: dict, total_episodes) -> None:
    line_width = 1.2
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}

    plt.figure(figsize=(10, 6), dpi=150)
    plt.ylim((-500.0, 0.0))
    plt.grid(c='lightgray')
    plt.margins(0.02)

    # Draw/remove axis lines
    for i, spine in enumerate(plt.gca().spines.values()):
        if i in [0, 2]:
            spine.set_linewidth(1.5)
            continue
        spine.set_visible(False)
    
    x = np.arange(total_episodes)
    plt.xscale('log')
    plt.xticks([1, 10, 100, 1000, 10_000, 100_000, 1_000_000], 
               ['1', '10', '100', '1000', '10,000', '100,000', '1,000,000'])
    
    for key, value in value_hist.items():
        # Assuming key is a string in the format "title,label"
        print(key)

        # Plotting the data with uniform filtering
        plt.plot(
            x, 
            uniform_filter(value, size=20), 
            linewidth=line_width, 
            label=key,
            c='tomato',
            alpha=0.95
        )

    plt.title(key + ' training record', fontdict=fontdict)
    plt.xlabel('Episodes (log scale)', fontdict=fontdict)
    plt.ylabel('Rewards', fontdict=fontdict)    
    plt.legend()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, f'{"_".join(key.lower().split())}.png')
    plt.savefig(save_path)
    plt.show()




def train_dqn(env, model, target_model, replay_buffer, episodes=1000):
    global epsilon
    
    for episode in range(episodes):
        state, _ = env.reset()
        state = np.array(state).reshape(1, -1)
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            step += 1
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(env.nA)  # Random action for exploration
            else:
                q_values = model(state)
                action = np.argmax(q_values.numpy())  # Exploit the learned policy

            # Take the action and get the next state and reward
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state).reshape(1, -1)
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            # Train the model if there are enough samples in the buffer
            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = train_step(batch, model, target_model)

            # Update the target model periodically
            if step % 100 == 0:
                target_dqn.set_weights(model.get_weights())

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

# Create the model and target model
action_space = 9  # The number of actions
dqn = DQN(action_space)
target_dqn.set_weights(dqn.get_weights())  # Initialize target model with same weights

# Initialize the replay buffer
replay_buffer = ReplayBuffer()

# Train the model
train_dqn(env, dqn, target_dqn, replay_buffer, episodes=1000)