# ReinforcmentLearning

Repository for different reinforcment learning projects.


## Background 

### Finite Markow Decision Process (MDP)
- Agent: Takes action
- Enviorment: From action produces rewards, and next states. Passed back to the agent.

Dynamics of the problem is given by p(s',r|s,a), provides the probailiity of next state and reward, given current state and action. 

**The Bellman Equations**

For any policy $\pi$, all $s \in S$ and all $a \in A(s)$: 
$$v_\pi (s) = \sum_{a \in{A(s)}} \pi(a|s)q_\pi(s,a)$$ (1)
$$q_\pi (s,a) = \sum_{a \in{A(s)} r \in R} p(s',r|s,a)[r + \gamma v_\pi(s')]$$ (2)


(1) the value of the state, when following the policy is equal to the policy weighted average of the action values. If we had 100 states, would needed 100 equations. (2) The action value of a state action pair is the probability weigthed average of the reward you'll get in the next step, and the discounted value of the next state.

Bellman equation for $v_\pi$ (state value function) is given by substion equation 2 into 1. 

$$v_\pi (s) = \sum_{a \in{A(s)}} \pi(a|s) \sum_{a \in{A(s)} r \in R} p(s',r|s,a)[r + \gamma v_\pi(s')]$$

Bellman equation for $q_\pi$ (action value function) is given by substion equation 1 into 2.

$$q_\pi (s,a) = \sum_{a \in{A(s)} r \in R} p(s',r|s,a)[r + \gamma \sum_{a' \in{A(s')}} \pi(a'|s')q_\pi(s',a')]$$


**Bellman Optimality Equations**

For any policy $\pi$, all $s \in S$ and all $a \in A(s)$: 

$$v_\pi (s) = max_{a \in A(s)} q_{\*} (s,a)$$ (1)
$$q_{\*} (s,a) = \sum_{a \in{A(s)} r \in R} p(s',r|s,a)[r + \gamma v_{\*}(s')]$$ (2)

One tweak, the state value must be the maximium over optimal action values. 

 **Policy Evaluation**
- States are randomly initialized
- These states are reffered to as V(s) because it estimates the true value.
- Reassigning  each state value according to the bellman equation
  
$$V(s) = \sum_{a \in{A(s)}} \pi(a|s) \sum_{a \in{A(s)} r \in R} p(s',r|s,a)[r + \gamma v_\pi(s')]$$

Intuion is that each assingment imports some small piece of the known MDP, making the value estimates a little better. For action value, process is similar excepet that states, are replaced by state action pairs.  

Meaning that state S wil be determined by states reachable from state S with actions A. Continue reassigning each state. One pass over all states is called a sweep. 

**Generalized Policy Iteration (GPI)**

Referes to entire class of algorithms. 

- Policy Iteration: To find $\pi*$ and $v_*$
  $$\pi_0 -E-> V_{\pi_0} -I-> \pi_1 -E-> v_{\pi_1}$$

Start with arbitrary initiliazed policy $\pi_0$, then apply policy evaulation which we determine it's value function. Then create a slightly better ploicy $\pi_1$. Now our value function no longer applies, then apply evaluation again to get a new function. Continues to no changing, arrived at optimal policy. 

**Value iteration** 




---



In episodic case, this takes place until some terminal state is reached. Goal is to determine a policy, which is a state dependet distribution over actions. Agent selects actions by sampling from this distribution. By running an agents policy through a MDP, we will get a trajectory (S0,A0, R1, S1, A1, R2, S2 ...) of states actions and rewards. Our goal is to achive a highest discounted sum of rewards (R), averaged over many trajectories. 

Dynamic programming and bellman equation possible when we have complete knowledge of the environment as an MDP, access to this p(s',r|s,a). 

An example for value based bellamn: given a state, the value of this state is the sum over the expected return given this state and the actions in this state, weighted/times the policy probability of choosing that action:

$$ \sum_a \pi(a|s^0) E_\pi[G_t|s^0,a] $$

G(t) obeys a simple recursive relationship: $G_t = R_{t+1} + \gamma G_{t+1}$. Replace $G_{t+1}$
with $v_\pi(S_{t+1}$. The value you get when plug the next state into the value function. This makes us able to rewrite: 

$$
E_\pi[G_t|S^0,a_0] = E_\pi[R_{t+1} + \gamma G_{t+1}|s^0,a_0] = E_\pi[R_{t+1} + \gamma v_\pi(S_{t+1} |s^0, a_0]
$$

Above we look at the excpected reward after taking action a_0 in state0, and replacing the $G_{t+1}$ as mentioned above. We treat Rt+1 and St+1 as random variables, calculating an expectaion. (Probability weigthed average of values) Find this in the action distribution over rewards.

Bellman equation idea is that if we can solve for some state values we can solve for all. It connects state values. 


