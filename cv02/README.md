# KIV / ANLP Exercise 02

*Deadline to push results:* 2024-10-27 23:59:59

*Maximum points:* 20+5

✅ 18

👍 5

❌ -2  💔,💔,💔,💔,💔 a pět zlomených srdcí...


------------------------------------------------------------------------

The Goal
========

Implement missing parts in the given template of the supervised machine
learning stack for estimating semantic textual similarity (STS). Train and
evaluate it on the given dataset. Use architecture described in the
following section.

What is Semantic Textual Similarity 
===================================

Semantic textual similarity deals with determining how similar two
pieces of texts are. This can take the form of assigning a score from 0
to 6 (Our data).

Project Structure 
=================

-   [data]
-   [tests]
    -   *anlp01-sts-free-train.tsv*
    -   *anlp01-sts-free-test.tsv*
-   *main02.py*


The Dataset
============

The dataset was generated during our collaboration with Czech News
Agency - so it is real dataset.

The training part of the dataset contains 116956 samples. Each sample
consists of two sentences and an annotation of their semantic
similarity.

    Příčinu nehody vyšetřují policisté.\tPříčinu kolize policisté vyšetřují.\t4.77

Tasks \[20+5 points in total\]
===============================

### Dataset Statistics **[1pt]**
Create histogram of pair similarity. 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![Alt text](./img/training_similarity.svg?raw=true "Classes in training dataset")
![Alt text](./img/testing_similarity.svg?raw=true "Classes in testing dataset")

(Test data pair similarity score is floating point, so the numbers were rounded in order to create a histogram)

(Which is weird to me, because training data pair similarity score is integer, so why do the test data have floats as score?)

We can clearly see from the histogram, that the dataset is not that balanced.
The training data consists of a lot of 0 scored pairs, while the test data have 0 score as the minority class.

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

Present mean and std of the dataset. 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

|       | Mean  | Std Dev |
|-------|-------|---------|
| Train | 2.447 | 2.119   |
| Test  | 2.660 | 1.810   |

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Baseline analysis **[2pt]**
What would the loss of a model returning a random value between 0 and 6 uniformly look like?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

The loss very much depends on concrete loss function used.

In my previous versions I rounded all the score  values to integers, because I assumed that the training data were integers and the test data were weird.

Now I am wiser and I know that the problem is not classification, but regression, so I was pretty stupid doing that.

I tested MSE (Mean Squared Error) loss function, as can be seen in the `baseline_analysis.py`, mainly because this loss function is actually being used in the real model later.

Results are as follows (results are averaged over 5 runs):

### Random Model

|       | MSE Loss (mean ± std dev) |
|-------|---------------------------|
| Train | 7.784 ± 0.024             |
| Test  | 6.230 ± 0.133             |

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

What would the loss of a model returning best prior (most probable output) look like?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Same text applies here, as above for the random model.

Results are as follows:

### Majority class Model

|       | MSE Loss |
|-------|----------|
| Train | 10.478   |
| Test  | 10.228   |

From the two tables above we can clearly see, that the random model worked better for the test data, because the majority class in training data is 0, whereas in the test data, the majority class was 1 (see the histograms above).

Both models have bad losses, because these models are not learning anything, their loss would be "constant" through the epochs and batches and time.

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Implement Dummy Model
1. **Analyze the Dataset**

    #### **CF\#1**

    Count occurrences of words in the datasset, and prepare a list of
    top\_n words

    **CKPT\#1**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I find it really odd that we are not using any lemmatization or stemming in this step.

Basically "Ahoj", "ahoj", "ahoj," "ahoj\n"... are all different words, which is not ideal in my honest opinion.

But unittest made me do it this way, so I did it.

UPDATE: After discussing this with You in the class, I understand that lower casing and lemmatization / stemming could even harm our models.

But I still don't like the fact that the score from the format "sentence\tsentence\tscore" (and the tabulators also) are not thrown away and are included in the vocabulary.

For this reason I have created a function called `dataset_vocab_analysis_better`, that I will be using.

I am still keeping the old function for tests; see `main02.py` and mentioned function for more commentary on this.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

2. **Prepare Word Embeddings**.
    https://drive.google.com/file/d/1MTDoyoGRhvLf15yL4NeEbpYLbcBlDZ3c/view?usp=sharing

    [//]: # (# https://fasttext.cc/docs/en/crawl-vectors.html)
    [//]: # (# EMB_FILE = "b:/embeddings/Czech &#40;Web, 2012, 5b tokens &#41;/cztenten12_8-lema-lowercased.vec")

    #### **CF\#2**

    Use the *list of top N words* for pruning the given Word embeddings.
    The cache will be stored on the hard drive for future use **CF\#3**.

    You should see two new files *word2idx.pckl* and *vecs.pckl*
    **CKPT\#2**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I would personally enjoy more description here, because what exactly is `wanted_vocab_size`?
Is that the size of the vocabulary with my utility tokens in mind (such as UNK and PAD)?
Or is that the number of known words, so the size of the vocabulary will be bigger than `wanted_vocab_size` after I add my tokens?

For example for `wanted_vocab_size = 20000` I personally put &lt;UNK&gt; and &lt;PAD&gt; on index 19998 and 19999, but I am not saying that the other solution, where &lt;PAD&gt; and &lt;UNK&gt; would be on 20000 and 20001 is wrong.
I just don't know which is the correct one.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

3. **Implement SentenceVectorizer** This module takes text as an input
    and transforms it into a sequence of vocabulary ids. **CF\#4**

    example:

    Input: Příčinu nehody vyšetřují policisté

    Output: 3215 2657 2366 7063

    **CKPT\#3**

4. **Implement the DataLoader** Implement a Python iterator that loads data from a text file and yields batches of examples. The iterator should handle a dataset stored in the text file and process it in batches as it iterates.

    Use implemented SentenceVectorizer to preprocess the data during
    initialization. **CF\#5** Implement the `__next__` function to
    return a batch of samples. **CF\#6** Shuffle training dataset after
    each epoch **CF\#7**

5. **Implement training loop**

    Implement basic training loop **CF\#8a**. Implement testing for
    model and dataset**CF\#8b**.

6. **Implement DummyModel** DummyModel computes the mean STS value of
    given dataset in the initialization phase and returns this value as
    a prediction independently on model inputs. **CF\#9**

> **[4pt]**

7. **Implement Neural Network Model**

    #### **CF\#10a** **CF\#10b**

    The Model takes two sequences of numbers as an input (Sentence A and
    Sentence B). Use prepared word embeddings (task-2) to lookup word
    vectors from ids. Add one trainable projection layer on top of the
    embedding layer. Use the mean of all words vectors in a sequence as
    a representation of the sequence.

    Concatenate both sequence representations and pass them through two additional fully-connected layers. The final layer contains a single neuron without an activation function. The output represents the value of the STS (Semantic Textual Similarity) measure.

    ![Visualization of the architecture from task
    7](img/tt-d.png)

8. **Implement Cosine Similarity Measure**

    Change similarity measure implemented with neural network from
    task-7 to cosine similarity measure (cosine of the angle between
    sentA and sentB).

    ![Visualization of the architecture from task
    7](img/ttcos.png)

> **[4pt]**

9. **Log these:**
     1. mandatory logged hparams `["random_emb", "emb_training", "emb_projection", "vocab_size", "final_metric", "lr", "optimizer", "batch_size"]`
     2. mandatory metrics : `["train_loss", "test_loss"]`
    
10. **Run Experiments with different Hyper-parameters** 

       1. Use randomly initialized embeddings/load pretrained.
   
           `random_emb = [True, False]`
       2. Freeze loaded embeddings and do not propagate your gradients into them / Train also loaded embeddings.
   
           `emb_training = [True, False]`
       3. Add projection layer right after your Embedding layer.

            `emb_projection = [True, False]`
       4. Size of loaded embedding matrix pruned by word frequency occurrence in your training data.
        
           `vocab_size = [20000]`
       5. Used final metric in two tower model. 
  
           `final_metric = ["cos", "neural"]`

11. **The best performing experiment run at least 10 times** **[2pt]**

12. **Tune More Voluntarily [0-5pt]**
Add more tuning and HP e.g. LR decay, tune neural-metric head, vocab_size, discuss

    **[5pt extra]**

# My results
## Hyper Parameter Analysis

### Parallel Coordinate Chart **[1pt]**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![Parallel Coordinate Chart](./img/parallel_coordinate_chart_all.svg?raw=true "Parallel Coordinate Chart")
(Please note that batch size is not in the chart, because my runs with batch size 1000 did not log batch size for some reason? Everything appears to be 500 batch size)

![Parallel Coordinate Chart](./img/parallel_coordinate_chart_best.svg?raw=true "Parallel Coordinate Chart")
(Parallel coordinate chart of the runs, that have test loss <= 2; those runs are the 5 chosen for the reproduction of 10 times)

Those best 5 runs have the following HPs in common:
- `batch_size = 500`
- `optimizer = "adam"`
- `lr_scheduler = "multiStepLR"`
- `lr = 0.01`
- `final_metric = "neural"`
- `emb_projection = True`

So the only difference between those 5 runs is in the `random_emb`, `emb_training` and `vocab_size`.

![Parallel Coordinate Chart](./img/parallel_coordinate_chart_best_10.svg?raw=true "Parallel Coordinate Chart")
(Parallel coordinate chart of the runs, that have test loss <= 2; after 10 runs repetitions)

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Table of my results **[4pt]**
1. list all tuned HP
2. add Dummy model into table
3. present results with confidence intervals (run more experiments with the same config)

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I tuned the following HPs:
1. `batch_size`:
    -  Tested values: 500, 1 000
2. `lr`:
    -  Tested values: 0.1, 0.01, 0.001, 0.0001, 0.00001
3. `optimizer`:
    -  Tested values: "sgd", "adam"
4. `lr_scheduler`:
    -  Tested values: "multiStepLR", "expLR"
5. `random_emb`:
    -  Tested values: True, False
6. `emb_training`:
    -  Tested values: True, False
7. `emb_projection`:
    -  Tested values: True, False
8. `final_metric`:
    -  Tested values: "cos", "neural"
9. `vocab_size`:
    -  Tested values: 20 000, 50 000

Side note, I named my runs with the convention of having all the parameters in the name, so I can now easily group runs by name and have the confidence interval nicely shown:

![Group by name](./img/test_loss_groupby_name_best_10.svg?raw=true "Group by name")

Raw data from wandb using group by name:

```
(config: mean ± std dev (number of runs))
not rand emb, not emb train, vocab 20k: 1.941 ± 0.055 (5 runs)
not rand emb, not emb train, vocab 50k: 1.947 ± 0.042 (5 runs)
not rand emb, emb train, vocab 20k: 1.882 ± 0.054 (4 runs)
not rand emb, emb train, vocab 50k: 1.862 ± 0.083 (6 runs)
rand emb, emb train, vocab 20k: 1.912 ± 0.061 (5 runs)
```
(note, number of runs is not 10, why? Because I filtered out the runs, that had test loss > 2)

(used confidence for confidence interval calculation is 95 %)

| MODEL CONFIG                                 | TEST LOSS ± CONFIDENCE INTERVAL |
|----------------------------------------------|---------------------------------|
| *Dummy model*                                | *3.230 ± 0.000*                 |
| *Random model*                               | *6.230 ± 0.117*                 |
| *Majority class model*                       | *10.228 ± 0.000*                |
| NOT(rand_emb)NOT(emb_train)(vocab_size=20k)  | 1.941 ± 0.048                   |
| NOT(rand_emb)NOT(emb_train)(vocab_size=50k)  | 1.947 ± 0.037                   |
| **NOT(rand_emb)(emb_train)(vocab_size=20k)** | **1.882 ± 0.053**               |
| **NOT(rand_emb)(emb_train)(vocab_size=50k)** | **1.862 ± 0.066**               |
| (rand_emb)(emb_train)(vocab_size=20k)        | 1.912 ± 0.054                   |

(Note: All models also have `batch_size = 500`, `optimizer = "adam"`, `lr_scheduler = "multiStepLR"`, `lr = 0.01`, `final_metric = "neural"`, `emb_projection = True`)

From that table we can clearly see, that the best models are the ones that have smaller batch size of 500, optimizer adam with learning rate of 0.01,
learning rate scheduler (LR Decay) of multiStepLR, final metric neural head and projection layer of embeddings.

The only differences are in whether the embeddings are randomly initialized or not, whether the embeddings are trainable or not and the size of the vocabulary.

The overall best model is the one with the pretrained embeddings (not random) and trainable embeddings. The vocabulary size does not seem to do much, but the overall best is the one with the bigger vocabulary of 50 000.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Discussion **[2pt]**

Which HP I tuned? 

Which had the most visible impact? 

Did I use another techniques for more stable or better results? 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

One concrete combination I just found before running the experiments on MetaCentrum will always raise this error:

```
Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The combination is `emb_training = False, emb_projection = False, final_metric = "cos"`.

The error even makes sense, because if we look at the architecture with this concrete setup, there is absolutely nothing to train.

For this reason there is an if statement in `main02.py` that will skip the backward pass if the model is of that exact combination setup.

Final grid search looks like this:

    "batch_size": [500, 1000],
    "lr": [0.1, 0.01, 0.001, 0.0001, 0.00001],
    "optimizer": ["sgd", "adam"],
    "lr_scheduler": ["multiStepLR", "expLR"],
    "random_emb": [True, False],
    "emb_training": [True, False],
    "emb_projection": [True, False],
    "final_metric": ["cos", "neural"],
    "vocab_size": [20000, 50000]

That is 1280 combinations, which is a lot, because I had to run everything at least a few times.

The grid search was being adjusted as it went, because I was trying to hit the <= 2 test loss, but I couldn't find the right combination.
(My first "prototype" of grid search was 512 combinations).

So I added `learning rate = 0.1`, which did not help at all.

So then I tried adding `batch size = 500`, which helped a lot and I reached the milestone of test loss being the value of 2.
Here it seems weird to me that there was predefined constant of `BATCH_SIZE = 1000` in the original code -- it seems like unnecessarily big number to me.
Sadly I realised I could change that way too late -- after about 2000 completed runs.
I blindly followed the original code, without thinking about changing batch size.
(For students to come after us, I would probably change this constant, or explicitly mention that the batch size is to be tuned too.)

Let's do a closer analysis for each of the HPs:

1. `batch_size`:
    - Sadly, this parameter was very crucial to get overall better results, no model with the original constant of 1000 had test loss <= 2.
    - My experiments only proved that the batch size of 1000 were way too big, and 500 performed better.
    - I would probably try to tune this parameter even more, because I think that the batch size of 500 is still too big; but I need to save some MetaCentrum Mana for later :) .
    - (I can't show a groupby chart here, because batch size 500 was so much better, that it's the only curve in the chart on the given scope by wandb)
    - But I really believe that this parameter is the most crucial to test more, because the initial constant of 1000 was overshot by a ton.
2. `lr`:
    - Overall best learning rate turned out to be 0.01 in the combination with the optimizer Adam.
    - ![Group by learning rate](./img/test_loss_groupby_lr.svg?raw=true "Group by learning rate")
    - (Note that the learning rate of 0.1 is not in the chart, because it was so bad, that it was not even in the scope of the chart)
    - 💔nice chart but no labels..

3. `optimizer`:
    - Adam performed the best with the learning rate of 0.01.
    - (Once again groupby chart would not show much here, because Adam just outperformed sgd so much, that it is out of scope)
4. `lr_scheduler`:
    - I personally did not have any assumptions about this parameter, because I did not know much about LR decay up until now.
    - ![Group by learning rate scheduler](./img/test_loss_groupby_lr_scheduler.svg?raw=true "Group by learning rate scheduler")
    - As we can clearly see, multiStepLR performed better than exponentialLR for this task.
    - 💔nice chart but no labels..
5. `random_emb`:
    - Here I personally thought, that the randomly initialized embedding would not be as good as the pretrained ones.
    - ![Group by random embedding](./img/test_loss_groupby_random_emb.svg?raw=true "Group by random embedding")
    - As we can see from the chart, the pretrained embeddings performed better than the randomly initialized ones, as expected by me.
    - (Note, be careful with trusting these charts, because as they say "Computing group metrics from first 50 groups", there is about 1300-1400 runs, but the chart is made from only 50 of those)
6. `emb_training`:
    - With this parameter, I personally expected the model to perform better with trained/trainable embeddings.
    - ![Group by embedding training](./img/test_loss_groupby_emb_training.svg?raw=true "Group by embedding training")
    - The chart above again supports my assumption.
    - 💔nice chart but no labels..
7. `emb_projection`:
    - I expected the model to be better with the projection layer, it just made sense this way.
    - ![Group by embedding projection](./img/test_loss_groupby_emb_projection.svg?raw=true "Group by embedding projection")
    - As we can see from the chart, my assumption was correct.
    - Not only from the chart, but also from the table above, all the best models had this parameter be True.
    - 💔nice chart but no labels..
8. `final_metric`:
    - Here I believed that cosine similarity would be really good, but I expected that the neural metric head would "learn" some hidden patterns between the vectors, something possibly better than just "angle" (?! if possible).
    - ![Group by final metric](./img/test_loss_groupby_final_metric.svg?raw=true "Group by final metric")
    - Here I somehow don't believe the chart above. I believe that the neural head should outperform cosine similarity, but the curve of cosine similarity is just constant.
    - I believe the last 50 runs might have been my mentioned combination which doesn't have anything to learn, so it just outputs "constant" loss (a bit different for each batch of course, but still, constant) -- so the chart might be lying to us here.
    - 💔nice chart but no labels..
9. `vocab_size`:
    - Here I thought that bigger vocabulary would be better, as the model "knows" more.
    - ![Group by vocab size](./img/test_loss_groupby_vocab_size.svg?raw=true "Group by vocab size")
    - Here my assumption was not really that correct, or we would need deeper analysis, bigger grid search with more options for this parameter.
    - But this parameter seems to have little to no effect on the test loss.
    - 💔nice chart but no labels..

![Parameter importance](./img/test_loss_parameter_importance.png?raw=true "Parameter importance")

Based on wandb "Parameter importance", learning rate was yet again the most important parameter.

In my opinion batch size was the second, but wandb doesn't think so, probably because I only tried values of 1000 and 500, and I think if we showed the wandb how lower numbers would perform, it would agree with my opinion on that.

Final metric seems to be also an important choice, as all the good models had the neural metric head.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

# To Think About:

## Practical Question

1.  Compare both similarity measures (Fully-connected, cosine):
    -   speed of convergence,
    -   behaviour in first epochs,
    -   accuracy.

2.  Does the size of the effective vocabulary affect the results? How?

3.  Have you sped up debugging the system somehow?

4.  Can we save some processing time in the final test (10 runs) without
    affecting the results?

5.  What is the role of UNK and PAD token in both models?

6.  Can you name some hints for improvement of our models?

7.  Can we count UNK and PAD into sentence representation average
    embedding? Does it affect the model?

8.  What is the problem with output scale with the neural network?

9.  What is the problem with output scale with cosine?

10. What is the best vocab size? Why? Task8

11. What is the best learning rate? Why?

12. Which hyper-parameters affect memory usage the most?

## Theoretical Questions

1.  Is it better to train embeddings or not? Why?

2.  Is it important to randomly shuffle the train data, test data? Why?
    When?

3.  What is the reason for comparing MSE on train dataset or testing
    dataset with Mean of training data or mean of testing data?

4.  Can you name similar baselines for other tasks (Sentiment
    classification, NER, Question Answering)?
