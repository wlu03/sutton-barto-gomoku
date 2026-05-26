# Main Idea
Reinforcement learning is different from supervised learning because it learns from evaluative feedback and not instructive feedback. In RL the agent is not told the correct action. It only receives a reward that evaluates how good the action was. 
- **Instructive Feedback**: Used in *supervised learning* and the learner is told the correct answer
- **Evaluative Feedback**: Used in *reinforcement learning* and the learner takes an action and receives a reward. 
## Exploration vs Exploitation
Since the agent only observes the reward from the action it took, it must try different actions to discover which is the best one. **Exploration** is trying new or uncertain actions to gain information. While **exploitation** is choosing the action that currently seems the best. 
# The Multi-Armed Bandit Problem
This is a simplified RL problem. You have several possible actions, called **arms**. Each arm gives a reward from an unknown distribution. The goal is to maximize total reward over time by learning which arm is best. 
```
Arm 1 -> unknown average reward
Arm 2 -> unknown average reward
Arm 3 -> unknown average reward
```
The agent repeatedly chooses arms and updates its estimates based on the rewards received. A bandit problem is called **non-associative** because there is only one situation. The agent does not need to learn different actions for different states. *It only needs to learn which action is the best overall.* 

Compared to full RL, the best action depends on the current state and future consequences. 
# 2.1 An n-Armed Bandit Problem
**Setup**: In an **n-armed bandit problem**, you repeatedly choose between $n$ different actions. Each action gives a numerical reward. The reward is random, but each action has its own fixed reward distribution. The goal is to maximize the total reward over many time steps. 
- **Each time step**: `choose an action -> receive a reward -> update your estimate`

## Sample Average Method
A natural way to estimate an action's value is to average all the reward received from that action. If action $a$ has been chosen $N_t(a)$ times before time $t$ and the rewards were $$R_1,R_2,...,R_{N_t(a)}$$then the estimated value is $$Q_t(a)=\frac{R_1+...+R_{N_t(a)}}{N_t(a)}$$This is just the average reward from choosing action $a$. The reason why this works is because if an action is selected more and more times, its average reward becomes closer to its true expected reward. In other words, as $$N_t(a)\rightarrow \infty$$ we get that $$Q_t(a)\rightarrow q(a)$$This is from the **law of large numbers**. The problem with [greedy action](#Appendix) selection is the fact is **always exploits**. The problem is that it never explores. So if an agent's early estimates are wrong, it can get stuck choosing bad actions.
## $\varepsilon$-greedy
The simple fix is called **epsilon-greedy** action selection. Most of the time, the agent chooses the greedy action, but with some small probability $\varepsilon$, it chooses randomly. $\varepsilon$-greedy keeps exploring. This means every action has a chance of being sampled. Over a very long time, each action will be selected many times. Because $\varepsilon$-greedy continues to explore forever, it will eventually learn the true value of every action. However, because it still explores with probability $\varepsilon$, it does not always choose the optimal action.
### Decaying $\varepsilon$ 
The idea of reducing epsilon is just to explore less later on. $$\varepsilon_t=\frac{1}{t}$$or $$\varepsilon_t=0.1\cdot 0.99^t$$
The advantage of $\varepsilon$-greedy depends on the task. The difference reward variances and how they affect choosing between the two: 
- **High Reward Variance**: If rewards are noisy, you want to explore 
- **Zero Reward Variance**: If rewards are deterministic, greedy performs well. Deterministic means each action always gives the same rewards. 
- **Nonstationary Environments**: Action values change over time. Exploration is needed forever. This is because the best action can change. Even if greedy found the best action earlier, it may become outdated later on. 
## Incremental Average Formula
 storing all previous rewards can be extremely taxing for memory. We only want to update and store the estimates a select number of times. The way to update the average without storing all the old rewards is to: $$Q_k, \;\text{be the avg of the first k-1 rewards}$$Then after receiving the $k$-th reward, $R_k$, the new average is $$Q_{k+1}=Q_k +\frac{1}{k}(R_k-Q_k)$$$$\text{New Estimate} = \text{Old Estimate} + \text{Step Size} \cdot \text{Error}$$
 
## Tracking a Non-stationary Problem
The sample-average method works well when. the problem is **stationary**. Many RL problems are **nonstationary** meaning that the true reward value of actions change over time. We should give **more weight to recent rewards** and **less weight to old rewards**. 

To track changing action values, we introduce a constant step size $$Q_{k+1}=Q_k +\alpha (R_k-Q_k)$$where $$0<\alpha \leq 1$$$\alpha$ is the step-size parameter or **learning rate**. 
### Exponential Recency-Weighted Average
$$Q_{k+1} = Q_k +\alpha (R_k - Q_k)$$Expand the RHS to $\alpha R_k + (1-\alpha)Q_k$. Substitute $Q_k$ with $$Q_{k} = \alpha R_{k-1} + (1-\alpha)Q_{k-1}$$So $$Q_{k+1}=\alpha R_{k} + (1-\alpha)\alpha R_{k-1} + (1-\alpha)^2 Q_{k-1}$$If you keep expanding you get that $$Q_{k+1} = (1-\alpha)^kQ_1 + \sum_{i=1}^k \alpha (1-\alpha)^{k-i} R_i$$
The current estimate is a weighted average of:
1. the initial estimate $Q_1$ 
2. all past rewards $R_1,R_2,R_3,...,R_k$ 
But recent rewards get larger weights. It's exponential because old rewards weights shrink by a power of $$1-\alpha$$For example, $\alpha = 0.1$ 
- $R_k=0.1$
- $R_{k-1}=0.1(0.9)=0.09$
- $R_{k-2}=0.1(0.9)^2=0.081$
So each order reward gets less weight. 
### Optimistic Initial Value
We might initialize all action values at zero $Q_1(a)=0$. What happens if we initialize all action values to something too high $Q_1(a)=5$. This is called an **optimistic initial value** because we are pretending every action is very good before we have evidence. Optimism encourages exploration because if every untried action looks good, the agent will naturally want to try them out. 

**For example**, if we initialize all estimates as $Q_1(a)=5$. But if the true action values are actually around $0$ where $q(a) \sim N(0,1)$. Most values are more likely to be closer to $0$ than $5$. At the start, every actions/arm is $Q=5$. The agent picks one action greedily and receives a reward off $R=0.8$. Since the reward is much lower than the estimate, the estimate for action 1 goes down. The agent is greedy, so next time it will choose one of the actions still estimated at 5. That means it tries another action. After that action also gets a reward below 5, its estimate goes down too. So the agent keeps moving to untried actions because untried actions still look best. This is why optimistic initial values create exploration **even with a greedy policy**.
## Upper Confidence Bound (UCB)
Explore actions that are uncertain and might be plausibly be good. The formula is $$A_t=\arg\max_a [Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}]$$This means that at time $t$, choose the action whose estimated value plus uncertainty bonus is largest. The uncertainty depends mostly on $N_t(a)$ where $$N_t(a)=\text{ number of times actions a has been selected before time t}$$If $N_t(a)$ is large, that makes the bonus become smaller. UCB uses the principle of **optimism under uncertainty**. The parameter $c$ controls exploration strength. 
- Large $c$ explores more
- Small $c$ explores less 
- UCB becomes greedy when $c=0$
## Gradient Bandits
All the methods previous were based on estimating $Q_t(a)$. This is saying how good is action $a$. Then, the agent selects actions using those estimates. But gradient bandits use a different idea. They do not directly estimate action values. Instead, they learn a **preference** for each action. A preference $H_t(a)$ is not the same as estimated reward. $Q_t(a)$ tries to estimate $$\mathbb{E}[R_t\mid A_t=a]$$$H_t(a)$ is just a score saying how much the agent prefers action $a$. Higher preference means that action is more likely to be selected. The preference can be any real number and does not equal reward. 
### Preference becomes Probabilities
Gradient bandits usually turn preferences into probabilities using softmax: $$\pi_t(a)=\frac{e^{H_t(a)}}{\sum_b e^{H_t(b)}}$$
This means actions with larger preferences have a larger probability of being chosen. Action-value methods estimate $Q_t(a)$ then choose based on those values. Gradient bandits directly learn a policy: $$\pi_t(a)$$through preferences $$H_t(a)$$
### Gradient Bandit Update
For the action selected, $A_t$ $$H_{t+1}(A_t)=H_t(A_t)+\alpha (R_t-\overline{R_t})(1-\pi_t(A_t))$$For actions not selected, $a\neq A_t$, they are updated too $$H_{t+1}(a)=H_t(a)-\alpha (R_t-\overline{R_t})\pi_t(a)$$
Where: 
- $\alpha$ is the step size
- $R_t$ is the reward
- $\overline{R_t}$ is an average reward baseline
- $\pi_t(a)$ is the probability of selecting action $a$
### Expected Reward Objective
The performance measure is that $$\mathbb{E}[R_t]=\sum_b \pi_t(b)q(b)$$$$\text{expected reward} = \sum_b \text{prob. of choosing b} \times \text{reward for choosing b}$$
where $q(b)$ is the true expected reward of action $b$ 
### Performance Gradient
Take the derivative with respect to preference $H_t(a)$. This is saying that, if I slightly change the preference of action $a$, how does the total expected reward change. 
$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \sum_b q(b)\frac{\partial \pi_t (b)}{\partial H_t(a)}$$
This is saying: How does the expected reward change if I change the preference for action $a$? 
    To get this, start with $$\mathbb{E}[R_t]=\sum_b \pi_t (b)q(b)$$Differentiate both sides with respect to $H_t(a)$ $$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \frac{\partial}{\partial H_t(a)} \sum_b \pi_t (b)q(b)$$Since $q(b)$ is treated as a constant $$... = \sum_b q(b) \frac{\partial \pi_t(b)}{\partial H_t(a)} $$
    $\frac{\partial \pi_t(b)}{\partial H_t(a)}$ tells us how changing the preference of action $a$ changes the probability of action $b$. For a softmax, there are two cases $b=a$ and $b\neq a$. In the first case when $b=a$ $$\frac{\partial \pi_t(b)}{\partial H_t(a)}  = \pi_t(a)(1-\pi_t(a))$$increasing $H_t(a)$ increase the probability of action $a$. In the second case when $b\neq a$ $$\frac{\partial \pi_t(b)}{\partial H_t(a)} = -\pi_t(b)\pi_t(a)$$increasing $H_t(a)$ decreases the probability of every other action. This can be combined where $$\frac{\partial \pi_t(b)}{\partial H_t(a)}  = \pi_t(b)(\mathbf{1}_{a=b} -\pi_t(a))$$$$\mathbf{1}_{a=b} = \begin{cases} 1, & a = b \\ 0, & a \neq b \end{cases}$$
[[Exercise - Gradient Bandit]] 

### Problems with $\mathbb{E}[R_t]$
In bandit learning, we don't know $q(b)$ for every action. At time $t$ you only observe on sample action and one reward. So instead of using the full sum over actions, we want a **sample-based estimator**. This is why we rewrite the gradient as an expectation. $$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E} [(R_t-\overline{R}_t)  \frac{\partial \pi_t(A_t)}{\partial H_t(a)} \frac{1}{\pi_t (A_t)}]$$
Instead the expectation, there are three main parts. 
- $R_t-\overline{R}_t$: This is the **reward advantage**. This is $$\text{current reward - average reward}$$This is better than using raw rewards alone because the same reward can mean different things in different environments. Thus, the baseline/average gives context to the current reward. 
- $\frac{\partial \pi_t(A_t)}{\partial H_t(a)}$: This is saying "How does the probability of the sampled action $A_t$ change if we change the preference of action $a$?" 
- $\frac{1}{\pi_t (A_t)}$: This appears because we are converting a full sum into a sample expectation. Because actions are sampled according to $\pi_t$, if we want to sample estimate of a sum over actions, we divide by the probability of the sample action. Rarely chosen actions need larger correction because they appear less often. 
Using the softmax derivative with $b=A_t$ $$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E} [(R_t-\overline{R}_t) (\mathbf{1}_{a=A_t} -\pi_t (a))]$$
### Sampled Update 
Since the exact expectation is unknown, we use the sampled quantity $$(R_t-\overline{R}_t)(\mathbf{1}_{a=A_t} - \pi_t(a))$$as an estimate of the gradient. The Gradient ascent update is: 
$$H_{t+1}(a)=H_t(a)+\alpha (R_t-\overline{R}_t) (\mathbf{1}_{a=A_t}-\pi_t(a))$$for every action $a$. This is gradient ascent, not descent, because we want to maximize reward. The learning rate $\alpha$ controls how big the update is. 
## Associative Search
Associative search adds **situations** or **contexts**. The best action may depend on the situation. The agent must learn what is the best action given the situation. So the goal becomes to learn a policy $$\pi(a\mid s)$$or $$\text{situation} \rightarrow \text{best action}$$
This is why it is called associative. The agent associates actions with situations.

# Appendix 
## Action Value $q_*(a)$
 Each action has a true expected reward. This is called the **value** of an action. $q_*(a)$ is the true mean reward of action $a$. The best action is the one with the highest value $$a^* = \arg \max_a q_*(a)$$
 If we knew every true action value, the problem would be easy by always choosing the action with the highest value. But in bandit problems, we **don't** know the true values. We only know the estimates. 
## Estimated Action Value $Q_t(a)$
Since the true action values are unknown, the agent keeps estimating. This is the **estimated action value**. This means, the agent's estimated value of action $a$ at time $t$. The agent updates $Q_t(a)$ based on the rewards it observes. 
## Greedy Action
A **greedy action** is the action with the highest current estimated value. $$A_t=\arg\max_a Q_t(a)$$For example, if we had the following estimated action values: 
- $Q_t(\text{Arm 1}) = 3$
- $Q_t(\text{Arm 2}) = 7$
- $Q_t(\text{Arm 3}) = 5$
The greedy action is Arm 2 because it currently has the highest estimate. Choosing the greedy action means you are using your current knowledge to get the highest expected immediate reward.
## Exploitation
When you are choosing the greedy action, you are **exploiting**. Exploitation is good for short-term reward because you are choosing the action that currently looks best.
## Exploration
When you are **not choosing the greedy action**, you are **exploring**. This means that you try actions that does not current look the best in order to learn more about different scenarios. Exploration may hurt short-term reward, but it can improve long-term reward.
