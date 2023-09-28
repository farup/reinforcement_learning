# Deep Qlearning 

Project with insipiration from [Patrick Loeber](https://www.youtube.com/watch?v=L8ypSXwyBds&t=4261s&ab_channel=freeCodeCamp.org)

### Introduction 

Reinforcement Learning (or RL) is a branch of Machine Learning where an agent learns to maximize the reward by interacting with the environment and understanding the consequences of good and bad actions. Goal is to find a sequence of actions that will maximize the return: the sum of rewards. 

In the very early beggining of RL most of the applications was Q-Tables. For each action at each state a maximum expected future reward was calculated. Based on this information, agent could choose the action with highest reward. But as the enviorments grows large, the amount of state-action pairs becomes infeasible to store in memmory. 


### Policy 
Policy is the mapping or function that tell us the action to take given a state. This is the function we want to learn. The optimal policy, that maximizes expected return when the agent acts according to it. Two approaches; 
- Policy-Based Methods: Teach the agent to learn which action to take, given the current state
- Value-Based Methods: Teach the agent to learn which state is more valuable, the take actions accordingly.

### Deep Q Learning
