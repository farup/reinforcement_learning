# Deep Qlearning 

Project with insipiration from [Patrick Loeber](https://www.youtube.com/watch?v=L8ypSXwyBds&t=4261s&ab_channel=freeCodeCamp.org)

### Introduction 

Reinforcement Learning (or RL) is a branch of Machine Learning where an agent learns to maximize the reward by interacting with the environment and understanding the consequences of good and bad actions. Goal is to find a sequence of actions that will maximize the return: the sum of rewards. 

In the very early beggining of RL most of the applications was Q-Tables. For each action at each state a maximum expected future reward was calculated. Based on this information, agent could choose the action with highest reward. But as the enviorments grows large, the amount of state-action pairs becomes infeasible to store in memmory. 

### Deep Q Learning

In Deep Q Learning, the agent uses the observations resulting from the selected action based on the Q-network to calculate a target Q-value for the current state-action pair. 

The Q-network predicts the quality or expected cumulative future rewards of taking a specific action in a given state. It estimates the Q-value, which stands for "quality" or "action-value," and represents the expected return (sum of rewards) an agent can achieve by taking a particular action from a particular state and then following a specific policy (action-selection strategy) thereafter.

### Snake 

Important code snippets from classes: 

**Agent**

```
class Agent:

    def get_state(self, game):
        """calculates the state of the game, array of size 11"""

    def get_action(self, state):
      ...
      if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
````
Get action either picks a random action (exploration) or by policy. 



**Model**
```
class QTrainer:
  def train_step(self, state, action, reward, next_state, done):
    ... # creating tensor of (n,m) shape unsqueezing
    pred = self.model(state) # Predict Q-values with current state
    
    target = pred.clone()
    for idx in range(len(done)): 
        Q_new = reward[idx]
        if not done[idx]:
            Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

        target[idx][torch.argmax(action[idx]).item()] = Q_new
```
Code snippet above shows train step function for estimating Q-values with the network, then calculating a target of the observed new state after taking previous action. Model is a simple feedforward neural network. 

```
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
```
Update step for after calculating the loss of between estimation and calculated target. 


### Policy 
Policy is the mapping or function that tell us the action to take given a state. This is the function we want to learn. The optimal policy, that maximizes expected return when the agent acts according to it. Two approaches; 
- Policy-Based Methods: Teach the agent to learn which action to take, given the current state
- Value-Based Methods: Teach the agent to learn which state is more valuable, the take actions accordingly.






