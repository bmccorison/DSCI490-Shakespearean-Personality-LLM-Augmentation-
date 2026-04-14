# Reinfocement Learning Strategies

The simplest approach for now would be a thumbs up/down approach, where each response is classified into 'chosen' and 'rejected' responses. This could then be implemented directly into the web interface or terminal tooling and logged for each evaluation and post-training. 

## Two Primary Approaches

There seems to be two approaches for implementing this:

### 1. Direct Preference Optimization (DPO) 

This can be easily implemented with the `DPOTrainer` package and essentially involves collecting **pairs** of good and bad responses from the model, then updating the model directly. This would likely be the simplest implementation, and it seems to be really good with smaller models.

Although there are plenty of ways that we could implement this, one way could be having the model generate a few different answers to the same prompt, then picking out the best and worst one for retraining. This could definitely help and potentially speed up post-training fine tuning. 

One issue with this implementation would be that we would not be able to do a thumbs up/down approach on singular responses, since this kind of retraining must occur in **response pairs**.

### 2. Reward Modeling + PPO

Although this is a significantly more complex approach than DPO, it would allow for training on singular responses and could potentially automate some RL. This can be broken up again into manual and automated approaches:

#### Manual Approach 

This would just involve rating responses on a thumbs up/down basis.

#### Automated Approach

This would involve training a smaller lightweight reward model (Something < 250M parameter), training this to be a classifier then feeding it into Huggingface's `RewardTrainer`. We can then use the `PPOTrainer` library to reinforce responses during training.

## Opinion on Optimal Approach

Due to time constraints, I believe that the DPO approach would be optimal due to the lower complexity and computation cost.
