import pdb
import numpy as np

from unityagents import UnityEnvironment

from agent import Agent
from utils import plot_learning_curve

seed = 123

env = UnityEnvironment("Tennis_Linux/Tennis.x86_64")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
num_actions = brain.vector_action_space_size
num_inputs = env_info.vector_observations.shape[1]

agent = Agent(
    n_inputs=num_inputs,
    n_actions=num_actions,
    n_agents=num_agents,
    random_seed=seed,
)

scores = []
thetas = []
avg_scores = []
best_score = -np.inf
game = 0
try:
    while len(avg_scores) < 100 or avg_scores[-1] < 1.0:
        game += 1
        score = np.zeros(num_agents)
        t = 0
        dones = [False] * num_agents
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        while not np.any(dones):
            action = agent.act(states, add_noise=True)
            env_info = env.step(action)[brain_name]
            states_ = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, action, rewards, states_, dones)
            states = states_
            score += rewards
            t += 1
            print(t, end="\r")
        scores.append(np.max(score))
        thetas.append(agent.noise.theta)
        avg_scores.append(np.mean(scores[-100:]))
        print(
            f"{t}, Eps {game:5d}: theta: {agent.noise.theta:0.2f}, avg: {avg_scores[-1]:6.4f}, last: {scores[-1]:6.4f}"
        )
        if avg_scores[-1] > best_score and game > 10:
            agent.save_checkpoint("checkpoints")
            best_score = avg_scores[-1]
except KeyboardInterrupt:
    pass

env.close()
plot_learning_curve(
    scores=scores,
    avg_scores=avg_scores,
    thetas=thetas,
    title="Learning Progess (Avg over agents)",
    figure_file="checkpoints/progress.png",
)
