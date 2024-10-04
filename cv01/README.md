# KIV/ANLP assignment 01

## Implement Training Loop and Experiment with Hyper-Parameters

## Prerequisities
1. Instal PyTorch (https://pytorch.org/)
2. Run Hello world (cpu/gpu)
3. Create account on MetaCentrum

## Tasks 

## Our team work  [0pt]

Complete missing parts and design clear interface for experimenting.
1. Use python argparser 
2. Use wandb and log everything
3. For easy login and testing use environment variable WANDB_API_KEY 
4. Run minimalistic hello world on MetaCentrum

## Individual work **[13pt in total]**

### Dataset Analysis **[1pt]**
Create histogram of classes in the dataset. 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![Alt text](./training_targets.svg?raw=true "Classes in training dataset")
![Alt text](./testing_targets.svg?raw=true "Classes in testing dataset")

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Baseline analysis **[1.5pt]**
How would look accuracy metric for **random model** and **majority class model**(returns only majority class as an output)

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Random model always returns a random class -- since the dataset is balanced (look above -- histograms), we can simply calculate the accuracy as 1 / number_of_classes.
Theoretically the random model should have around 10 % accuracy, since there are 10 classes in the MNIST dataset

Majority class model always returns the majority class -- from the histograms above, we can clearly see that the majority class is the class 1.
Since the dataset is almost well balanced, we can assume that the accuracy is going to be something around 10 % too (I'm expecting a little over 10 %, 
since the class 1 is majority class, so maybe 10.5 %, or even 11 %)

I've implemented this in the `baseline_analysis.py` script and the results from the script are as follows:
- Random model accuracy: 0.1000 (on average -- of course this is seed dependent)
- Majority class model accuracy: 0.1135 (this number is always the same, obviously)

![#008000](https://placehold.co/16x16/008000/008000.png) `Answer end`

Is there any drawback? Can we use something better, why?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Main drawback is that those accuracies are not very good -- those models are not very good

Those models are good for baseline comparison -- when I have some great new model, I can see if it truly is that great
based on comparing it with those models. If accuracy of my model is worse than random model (or majority class model),
then the awesome model isn't that awesome after all

Of course we can use something better -- for example some simple neural network, or even better, some pre-trained models, LLMs etc.

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

1. Implement missing fragments in template main01.py
2. Implement 3-layer MLP with ReLU activation function **CF#Dense** 
3. Run Experiments **[3pt]**
   1. Run at least 5 experiments with all possible combinations of following hyper-parameters 
   2. Draw parallel coordinates chart and add image output into output section in this README.md

            `model: ["dense", "cnn"]`
            `lr: [0.1, 0.01, 0.001, 0.0001, 0.00001]`
            `optimizer: ["sgd","adam"]`
            `dp: [0, 0.1, 0.3, 0.5]`

   Each experiment train at least for 2 epochs.

4. Utilize MetaCentrum **[3pt]**

   For HP search modify attached scripts and utilize cluster MetaCentrum. 
https://metavo.metacentrum.cz/

# My results
## Parallel Coordinate Chart with Appropriate Convenient Setup **[0.5pt]**
Draw parallel coordinate chart with all tuned hyper parameters

1. Show all your runs **[0.5pt]**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![Alt text](./parallel_chart_all_hyper_params.svg?raw=true "Parallel Coordinate Chart with all Tuned Hyper Parameters")

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

2. Show only runs better than random baseline. **[0.5pt]**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

(my baseline is random model, so only models with test accuracy >= 10 % are shown)
![Alt text](./parallel_chart_all_hyper_params_better_than_random_model.svg?raw=true "Parallel Coordinate Chart with all Tuned Hyper Parameters")

(another baseline was chosen -- only models that have 95% accuracy or more are shown)
![Alt text](./parallel_chart_all_hyper_params_better_than_95_acc.svg?raw=true "Parallel Coordinate Chart with all Tuned Hyper Parameters")

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Table of my results **[1pt]**
1. show 2 best HP configuration for dense and cnn model 
(both configurations run 5 times and add confidence interval to the table)
2. add random and majority class models into the result table
3. mark as bold in the table

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

My 2 best cnn runs were `genial-sky-437` (opt=sgd, lr=0.01, dp=0.5) and `expert-dust-159` (opt=adam, lr=0.001, dp=0.5)

My 2 best dense runs were `denim-paper-195` (opt=sgd, lr=0.1, dp=0.5) and `super-energy-150` (opt=adam, lr=0.001, dp=0.5)

(the values were taken from the wandb using filters and scalar chart (mean + std dev))

```
(mean ± std dev (number of runs))
98.027 ± 1.359 (7 runs)
98.360 ± 0.719 (9 runs)
93.512 ± 2.035 (8 runs)
95.646 ± 0.6406 (7 runs)
```

(used confidence for confidence interval calculation is 95 %)

| MODEL CONFIG           | ACCURACY ± CONFIDENCE INTERVAL |
|------------------------|--------------------------------|
| CNN_SGD_0.01_0.5       | 98.027 ± 1.007                 |
| **CNN_ADAM_0.001_0.5** | **98.360 ± 0.470**             |
| DENSE_SGD_0.1_0.5      | 93.512 ± 1.410                 |
| DENSE_ADAM_0.001_0.5   | 95.646 ± 0.475                 |
| RANDOM                 | 10.000 ± 0.000                 |
| MAJORITY CLASS         | 11.350 ± 0.000                 |

(bold marked model is the overall best model)

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Present all konvergent runs **[0.5pt]**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

These are not "all" the runs, but those are about 50 runs that had exceptionally good test accuracy -- higher than 98 %

![Alt text](./convergent_test_acc.svg?raw=true "Convergent runs")

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

Let's have a closer look on one concrete run -- `wobbly-valley-624` (model=cnn, opt=adam, lr=0.001, dp=0.1)

On the next graphs we can see that the model is so good, that the smoothing of curves makes it look worse than it truly is -- that's an exceptionally good run

![Alt text](./wobbly_valley_acc.svg?raw=true "Wobbly valley accuracy")

![Alt text](./wobbly_valley_loss.svg?raw=true "Wobbly valley loss")

## Present all divergent runs **[0.5pt]**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

These are not "all" the runs, but those are about 50 runs that had the test accuracy lower than 20 %

![Alt text](./divergent_test_acc.svg?raw=true "Divergent runs")

If we look at the threshold of about 10 %, we can see a lot of models concentrated around that value and "converged" to it, because they simply learnt to predict either the majority class or a random class

However, we can see some models were trained to be utterly bad -- even worse than random guesses, let's have a closer look at two of them -- concretely `divine-water-470` (model=dense, opt=sgd, lr=0.00001, dp=0.3) and `clear-terrain-489` (model=dense, opt=adam, lr=0.1, dp=0.1)
- `divine-water-470`: on closer look, this model doesn't appear to be truly divergent, the model only has really low learning rate and he didn't have enough time to learn. The trends in the graphs are not bad, but the scale of Y axis is really small -- the model needs bigger learning rate value

![Alt text](./divine_water_acc.svg?raw=true "Divine water accuracy")

![Alt text](./divine_water_loss.svg?raw=true "Divine water loss")

- `clear-terrain-489`: this model is truly divergent -- the accuracy started off better, than it ended -- the model "diverged" to about 10 %, thus becoming a random model
   - I am not entirely sure this is "divergent" run, because the model "converged" to something, but I put it here, because it's interesting

![Alt text](./clear_terrain_acc.svg?raw=true "Clear terrain accuracy")

![Alt text](./clear_terrain_loss.svg?raw=true "Clear terrain loss")

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Discussion **[1pt]**
- Discuss the results. 
- Try to explain why specific configurations work and other not. 
- Try to discuss the most interesting points in your work. 
- Is there something that does not make any sense? Write it here with your thoughts. 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Generally speaking, we can assume that optimizer `adam` works better with lower learning rates, while `sgd` works better with higher learning rates

From my 2 best runs for each model, we could assume that the networks with higher dropout probability had overall better results.
This assumption would be wrong according to this graph

![Alt text](./groupby_dp.svg?raw=true "Test Accuracy - Group by Dropout Probability")

As we can see from the graph, lower dropouts have bigger variance, thus allowing for randomly exceptional runs, whereas higher dropouts tend to have lower variance -- bigger dropout is more stable, but on average worse.
Which seems weird to me, because my top 2 runs for each model had the biggest possible dropout.

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

On the next two pictures we can see the importance of hyper parameters in relation to accuracy and loss

![Alt text](./parameter_importance_acc.png?raw=true "Importance of hyper parameters - accuracy")

![Alt text](./parameter_importance_loss.png?raw=true "Importance of hyper parameters - loss")

We can clearly see that the learning rate is the most important hyper parameter, followed by optimizer

## Try to explain why specific configurations works better than others. 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Certain configurations work better, because maybe the hyper parameter combination is more suitable for the specific problem and model
- for example as stated above, it seems like `adam` works better with lower learning rates, while `sgd` works better with higher learning rates

As shown above (importance of hyper parameters), learning rate is the most important hyper parameter, thus choosing the right learning rate value is crucial for the specific configurations to work well

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

# Something to think about

1. How to estimate the batch size?
2. What are the consequences of using a larger/smaller batch size?
3. What is the impact of batch size on the final accuracy of the system?
4. What are the advantages/disadvantages of calculating the test on a smaller/larger number of data samples?
5. When would you use such a technique?
6. How to set the number of epochs when training models?
7. Why do the test and train loss start with similar values? Can initial values have any special significance?
8. Is there any reason to set batch_size differently for train/dev/test?
9. When is it appropriate to use learning rate (LR) decay?
