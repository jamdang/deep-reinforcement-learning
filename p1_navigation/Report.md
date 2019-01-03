[//]: # (Image References)

[image1]: score-v-episode.png  "scores during training"

# Project 1 Report: Navigation

### Learning Algorithm

To train the agent, a [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) approach is employed in its original form to solve the navigation and banana picking task for this project.

Two Q-networks are used, one for the actual action value function approximation, called `qnetwork_local` (in `dqn_agent.py`); one as the target Q-network to provide a more stable target value function estimation, this one called `qnetwork_target`. Both Q-networks have the same architecture, as defined in `model.py`, i.e., a fully connected neural network with 3 hidden layers, each with 64, 36, 10 neurons, respectively. (The numbers of neurons in the input and output layers correspond to the dimensions of the state space and action space, i.e., 37 and 4.)

For every `UPDATE_EVERY` steps, we update both Q-networks, but in different fashions (as opposed to update every step). For `qnetwork_local`, we calculate the target value and form the loss and do the back-propagation to get the gradient and take a step in the gradient descending direction in the parameter space, the usual supervised-learning style. For `qnetwork_target`, we do a soft update for the parameter, i.e., using a first-order discrete filter to slowly update the network parameters towards that of `qnetwork_local`. The coefficient of the filter used for the soft update is chosen to be 0.001.

Another trick of DQN is the replay buffer that serves as a (fixed-sized) pool of past experience in the form of tuple `(state, action, reward, next_state, done)` which, when randomly sampled from, decorrelate the training samples for Q-network parameter update. The buffer size used for this project is 100,000, while the sample batch size used is 64.

Other hyperparameters and their values can be found in the beginning of the file `dqn_agent.py`.

### Plot of Rewards

The score of each episode during the training is plotted as follows.  

![score v.s episode][image1]

With the hyperparameters and the NN architecture described in the above section, the environment can be solved in 741 episodes	(The average score to solve the environment is defined to be 16.01).
At the end of `Navigation.ipynb`, we test the trained agent for three episodes and got the sampled scores of 10.0, 19.0, 19.0.

### Ideas for Future Work

For future improvement, any extension to the original DQN as described in the "Extensions to DQN" section of the [Rainbow Paper](https://arxiv.org/pdf/1710.02298.pdf) can be tried out.
