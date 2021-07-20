# Project Report

## Implementation Details
### agent.py
This file implements the DDGP Agent. DDGP is a actor-critic reinforcement learning algorithm. In a nutshell, the critic tries to constantly estimate the achievable reward based on the current state, while the actor tries to find a policy that makes the critic predict the highest scores.

The agent class holds the Actor and Critic models and implements functions for choosing an action for a given state, training the actor and critic models as well as loading and saving checkpoints.\
For the size of the replay buffer and batch size common default values were chosen with 100k, 256.\
For learning rates and the soft-update ratio tau I went a bit more aggressive than in project 2 to improve the learning rate. Tau was set to 0.05 and the learning rate to 0.001.
Gamma was set to 0.99. A pretty common value for tasks in which rewards are given rather sparsely.\
Since I collected the experiences of two agents in a shared replay buffer, I decided to run train the agent twice at every second timestep.

### memory.py
The ReplayBuffer class is defined in this file. It is little more than a larger FiFo buffer that can be sampled for past experiences.

### noise.py
This file implements the OUNoise class. During training, this noise is added to the chosen action to ensure a good expoloration of the state space and to not get trapped in a local minimum immediately.

The [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) is used to generate noise that is controllably decaying over the course of the training. \
There are three parameters that describe the process:
* `mu`: This is the expected mean value
* `sigma`: The standard deviation
* `theta`: How fast the process reverts to mu

At the beginning of the training, we start off with a very low theta to allow high noise values for longer to boost expoloration. Over the course of one epoch, the amount of noise then reduces gradually due to the mean-reverting property of the OU-Process. \
After each epoch, the process is resetted and theta is increased by `theta_incr` until it eventually reaches a value of one. At this point the amount of noise is only given by `mu` and `sigma`.

The graphic below shows the trajectory of the OU-Process for different theta values. Note that the starting point for each trajectory is chosen randomly between -1 and 1.

![Noise Decay](noisedecay.png)

### model.py
Actor and Critic models are defined in this file. \
The actor is a simple 2-layer network with 128 and 65 hidden nodes (65 because apparently I can't type). \
The critic is a 2-layer network where the actor's output is fed directly into the first hidden layer. The layers have 130 (128 + 2 actions) and 64 nodes, respectively.\
Both models use the ReLU activation function exclusively and feature a batch normalization layer after each hidden layer which greatly improved performance.

### utils.py
Utility functions are defined in this file. Currently there is only one:
* `plot_learning_curve`: takes a list of scores, a list of average scores, a list of theta values, an output path and a title and creates the progress like this one:
![progress](checkpoints/progress.png)

### train.py
This is the entrypoint for training a model. It does not require any arguments and will simply train a model until the average score over 100 episodes reaches 1.0.

### test.py
This is the entrypoint for testing a trained model. It takes the path to the checkpoint folder as argument and will run the single-agent environment for 10 episodes. The achieved score is printed to the terminal after each episode.

## Potential Improvements
I did not try to optimize pretty much any of the hyperparameters so there might be still room for improvement. The training speed could probably also be improved by running multiple tennis matches in parallel.

