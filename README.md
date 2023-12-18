This project outlines sampling strategies for eliciting voters preferences with the ultimate goal of computing a Kemeny ranking. 
Note that the sampling strategies may also be used for eliciting other aggregate rankings of voter preferences based on C2 ranking functions 
(however, without the same guarantees). 

This code was used to produce the experimental results in the AAAI 2024 paper on "Eliciting Kemeny Rankings" by Anne-Marie George and Christos Dimitrakakis, to be published by AAAI press 2024.
It consists of the following main components:

* Functions to generate preference profiles of specified size uniformly at random and to build the corresponding preference matrices.  For reproducability, randomness is controlled by random seed assignment.
* An ILP formulation to compute Kemeny rankings.
* Functions for computing confidence intervals (based on results in the paper).
* The "pruning" method to prune confidence intervals as outlined in the paper in Algorithm 3.
* Sampling strategies for uniform, opportunistic, optimistic, pessimistic and realistic (ie Bayesian) sampling w.r.t. pruned confidence bounds in case pruning is required.
* An overall elicitation scheme with choice of sampling strategy, pruning of confidences and sampling with or without replacement.
* The experiments setup, including plots etc.
