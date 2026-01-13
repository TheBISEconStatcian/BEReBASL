# Brainstorming

## How to make the policy training

In general we can consider the following set up for clients of this methodology:

- The client is an existing financial institution already suffering from rejection bias
- The client has kept the data of the applicants including the ordering in which they have applied

For this case one could train an agent for the policies in BASL (specially in Bayesian Evaluation) with this method:

Train + Evaluation based on passed data (all but last round), evaluate model again with respect to the last round. Differences in the evaluations a. k. a. expectations on the model are important and we can aim to reduce them through bayesian evaluation. A good way for training the RL problem would be this differences which are observable: how was I expecting that my model would perform vs how did it really perform on the last realized round.