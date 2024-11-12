import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import uniform_filter
import logging
from race_track_RL_env import RaceTrack
import os 

matplotlib.use('TkAgg')

# Configuration du logging
logging.basicConfig(level=logging.INFO)


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


# Off-policy Monte Carlo importance sampling algorithm
def off_policy_monte_carlo(total_episodes: int, track_map: str, render_mode: str) -> tuple:
    gamma = 0.9
    epsilon = 0.2
    epsilon_decay = 0.99999
    min_epsilon = 0.01

    env = RaceTrack(track_map, render_mode, size=100)
    action_space = env.nA  # (9, ), nine actions in total
    observation_space = env.nS  # (curr_row, curr_col, row_speed, col_speed)

    Q = np.random.normal(size=(*observation_space, action_space))
    Q -= 500  # optimistic initial values
    C = np.zeros_like(Q)
    target_pi = np.argmax(Q, axis=-1)

    reward_hist = np.zeros(shape=(total_episodes), dtype=np.float32)

    for i in tqdm(range(total_episodes), desc='Training', unit='episode'):

        trajectory = []
        terminated = False
        state, info = env.reset()
        (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)
        
        ttl_reward = 0

        # Generate a trajectory using behavior policy
        while not terminated:
            observation, reward, terminated, _ = env.step(action)
            
            ttl_reward += reward
            trajectory.append((state, action, reward, act_prob))
            state = observation
            (action, act_prob) = behavior_pi(state, env.nA, target_pi, epsilon)
        
        G = 0.
        W = 1.
        # Loop inversely to update G and Q values
        while trajectory:
            (state, action, reward, act_prob) = trajectory.pop()
            G = gamma * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])

            target_pi[state] = np.argmax(Q[state])
            if action != target_pi[state]:
                break
            W *= (1 / act_prob)
        
        reward_hist[i] = ttl_reward
        
        # Mise à jour d'epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    return reward_hist, Q

def simulate_optimal_episode(env, Q, render=False):
    """
    Simulate a single episode using the learned policy (Q-table).
    
    Parameters:
    env : the environment to simulate the episode on
    Q : the learned Q-table
    render : whether to render the episode or not (default: False)
    
    Returns:
    total_reward : the total reward accumulated during the episode
    """
    state, info = env.reset()  # Reset environment to initial state
    terminated = False
    total_reward = 0
    trajectory = []
    
    while not terminated:
        if render:
            env.render()  # Show environment if required
        
        # Get the best action from the Q-table
        action = np.argmax(Q[state])  # Choose action based on learned policy
        
        # Take the action in the environment
        next_state, reward, terminated, _ = env.step(action)
        
        total_reward += reward
        trajectory.append((state, action, reward))  # Store the trajectory
        state = next_state  # Move to the next state

    if render:
        env.render()  # Show the final state
    print(f"Total reward for this episode: {total_reward}")
    
    return total_reward, trajectory

if __name__ == "__main__":
    train = False # Switch between train and evaluation
    track_sel = 'a'
    total_episodes = 10000

    if train:
        reward_hist_dict = dict()
        Q_dict = dict()

        track_name = f'Track {track_sel.capitalize()}'
        key = track_name

        reward_hist, Q = off_policy_monte_carlo(total_episodes, track_sel, None)
        reward_hist_dict[key] = reward_hist
        Q_dict[key] = Q
        
        plot_result(reward_hist_dict, total_episodes)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path2 = os.path.join(script_dir, f'track_{track_sel}.pkl')
        with open(file_path2, 'wb') as f:
            pickle.dump(Q_dict, f)
            print("Q values saved")

    else:  # Evaluate the Q values and plot sample paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path2 = os.path.join(script_dir, f'track_{track_sel}.pkl')
        with open(file_path2, 'rb') as f:
            Q_dict = pickle.load(f)

        key = list(Q_dict.keys())[0]
        Q = Q_dict[key]
        policy = np.argmax(Q, axis=-1)  # greedy policy
        
        env = RaceTrack(track_sel, 'human', 5)
        fig = plt.figure(figsize=(12, 5), dpi=150)
        fig.suptitle('Sample trajectories', size=12, weight='bold')

        for i in range(10):
            track_map = np.copy(env.track_map)
            state, obs = env.reset()
            terminated = False
            while not terminated:
                track_map[state[0], state[1]] = 0.6
                action = policy[state]
                next_state, reward, terminated,_= env.step(action)
                state = next_state

            ax = plt.subplot(2, 5, i + 1)
            ax.axis('off')
            ax.imshow(track_map, cmap='GnBu')
        plt.tight_layout()

        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(script_dir, f'track_{track_sel}_paths.png')

        plt.savefig(save_path)
        plt.show()

