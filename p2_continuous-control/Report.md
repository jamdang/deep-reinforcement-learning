[//]: # (Image References)

[image1]: score-plot.png  "scores during training"

[image2]: DDPG-BD.png  "DDPG block diagram"

# Project 2 Report: Continuous Control

### Learning Algorithm

To train the agents, a [DDPG](https://arxiv.org/abs/1509.02971) approach is employed with some slight modifications to solve the Reacher environment for this project.

DDPG is an off-policy, actor-critic deep reinforcement learning approach, with the architecture depicted in the figure below.

![ddpg diagram][image2]

The actor &#956;(s) takes in the state s and tries to generate the optimal (continuous) action vector, while the critic Q(s,a) evaluates the action value at the state-action pair (s,a). Both the actor and the critic are represented by deep neural networks, with parameter &#952; and &#969;, respectively. To train a DDPG agent, at each step, we update the Q network (critic) to minimize the difference between the action value predicted by the local Q network and that obtained from one-step look-ahead of the target Q network (the discounted TD(0) action-value + reward); and we update the actor network in the direction that would maximize the Q-value given by the (local) Q-network.

As can be seen, the DDPG approach is in some way like the [DQN](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) approach and sometimes called an extension to DQN, although the latter is used for RL problems with discrete action space and DDPG is applied to continuous problem. Two of the DQN tricks are also used in DDPG:
1. The use of target networks, and
2. The use of a replay buffer  

During training, we maintain a target network for the actor and a target network for the critic, i.e, we have two copies of the actor network (with the same architecture, called `actor_local` and `actor_target`), and two copies of the critic network (also with the same architecture, `critic_local` and `critic_target`), all defined in `ddpg_agent_20.py`. The network architectures are defined in `model.py`. For the actor, a fully connected neural network is used, with 2 hidden layers, each with 128, 64 neurons, respectively (The numbers of neurons in the input and output layers correspond to the dimensions of the state space and action space, i.e., 33 and 4 for this project). For the critic, we also use a fully connected neural network. We first transform the incoming state vector with one-layer perceptron, with 128 neurons, and then we concatenate the result of the transform with the action vector, we then feed this concatenated tensor through two more hidden layers, each with 64 and 20 layers, before the output layer, which gives the scalar action-value.

Another trick of training a DDPG agent is the use of a replay buffer that serves as a (fixed-sized) pool of past experience in the form of tuple `(state, action, reward, next_state, done)` which, when randomly sampled from, decorrelate the training samples for Q-network parameter update. The buffer size used for this project is 1000,000, while the sample batch size used is 128.

Some other "tricks" employed in this project worth noting include:
- For each step, we collect the experience at this step from all 20 agents and put them in the (shared) replay buffer
- We update every `UPDATE_EVERY` steps (chosen to be 10 for this project), and every time we update we update `UPDATE_TIMES` times (chosen to be 20 for this project) + 1, if there are enough many samples in the replay buffer
- We use a soft update to update the target networks. The coefficient of the filter used for the soft update is chosen to be 0.001
- During training, we use some sort of Epsilon-greedy approach for action selection (instead of using added noise to the actor output) to guarantee certain level of exploration. We chose `eps` to be 0.1, i.e., with probability 0.9 we use the action from `actor_local`, while with probability 0.1 we use some totally random action.

Other hyperparameters and their values can be found in the beginning of the file `ddpg_agent_20.py`.

### Plot of Rewards

The score of each episode during the training is plotted as follows.  

![score v.s episode][image1]

With the hyperparameters and the NN architecture described in the above section, the environment was solved in 112 episodes	(The average score to solve the environment is defined to be 32.0).
At the end of `Continuous_Control_20.ipynb`, we test the trained agents and got a sample average score of more than 38.0.

### Ideas for Future Work

For future improvement, any extension to the original DQN as described in the "Extensions to DQN" section of the [Rainbow Paper](https://arxiv.org/pdf/1710.02298.pdf) can be tried out.
