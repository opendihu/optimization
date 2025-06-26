## Options for Bayesian Optimization
Bayesian Optimzation can optimize very diverse functions depending on its parameters. These parameters can be adjusted in the file ```setup_BayesOpt_general_1D.py```. 

- The options for the kernel are the RBF-kernel and the Matérn kernel with $\nu \in$ {0.5, 1.5, 2.5}.
- The possible mean functions are the constant mean and the zero mean.
- The noise can be chosen fixed or variable.
- The implemented acquisition functions are the Expected Improvement (EI), Probability of Improvement (PI), Knowledge Gradient (KG) and Entropy Search (ES).
- The rest are minor changes like the stopping criterion or the initial values

## General recommendation
The discussion about the best parameter setup for every case can be complicated and there is generally no best option. There has been a Bachelor thesis that discusses the best parameters for nice functions. The corresponding GitHub repository can be found [here](https://github.com/opendihu/optimization/releases/tag/Bachelor-thesis). The general recommendation of this Bachelor thesis is the Matérn kernel with $\nu$=0.5, the constant mean function, fixed noise, the ES acquisition function and the xy-stopping criterion. But this recommendation is just for the general case of nice functions. If we have a special function of which we have some information, there might be a better option.

## Biceps case
There is the [isometric biceps case](../opendihu_examples/isometric_contraction/biceps_muscle/) for which we want to discuss special parameters. If we use the recommended parameters, we get the following optimization plot: 
![](../figures/isometric_biceps.png)
We have found a maximum this way, which is good, but there are some disadvantages of this optimization process.
- There are very large areas of high uncertainty in which we don't know if we might have a hidden maximum
- The mean function is not very smooth and the trials are in local maxima, which very realistic

The question now is, if there is a parameter setup that finds the same (or a better) maximum without those disadvantages.

#### Entropy Search
The recommended setup with the Matérn kernel and $\nu$=0.5 is aparently working fine, but not optimally. If we change $\nu$ to 1.5 or 2.5, the result is immediately much worse. Since $\nu$ is the smoothness parameter, a higher value changes the assumed smoothness of the function. This has two disadvantages. The first one is the compatibility with ES. This acquisition function is working with minimizing entropy, which is much easier if we have more uncertainty. We lose that with a higher smoothness parameter. The second disadvantage is that we might not find the maximum in this case. The maximum is just at the edge of a high jump downwards. With a high smoothness the mean function will interpolate the trials in a way that will not allow high jumps, so we might look for a maximum elsewhere.

However, there is the option of adjusting the ES acquisition function so that we at least remove the large areas of high uncertainty. An option for this would be an evaluation of our function every 3 or 4 evaluations at the value with the highest uncertainty. In the case above this would add about 2 evaluations. That can have two effects. Either this results in a new and better maximum, or we get a nicer picture in the end. Keeping in mind that an evaluation takes an hour, one has to evaluate if this is worth the risk. Since you can always do just one evaluation more after the optimization process is done, the acquisition function should probably not be changed (if we're using ES), and depending on the result we can add more evaluations in areas we deem necessary.

#### Expected Improvement
