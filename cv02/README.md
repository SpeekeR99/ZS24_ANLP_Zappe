# KIV / ANLP Exercise 02

*Deadline to push results:* 2024-10-27 23:59:59

*Maximum points:* 20+5

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

UPDATE: For whatever reason, the function took really long time, but when I swapped one list for a dictionary, it is unbelievably faster.

Big "WHAT" moment for Python.

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

_MISSING_

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Table of my results **[4pt]**
1. list all tuned HP
2. add Dummy model into table
3. present results with confidence intervals (run more experiments with the same config)

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

_MISSING_

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Discussion **[2pt]**

Which HP I tuned? 

Which had the most visible impact? 

Did I use another techniques for more stable or better results? 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

One concrete combination I just found before running the experiments on MetaCentrum will always raise this error:

```python
Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The combination is `emb_training = False, emb_projection = False, final_metric = "cos"`.

The error even makes sense, because if we look at the architecture with this concrete setup, there is absolutely nothing to train.

For this reason there is an if statement in `main02.py` that will skip the backward pass if the model is of that exact combination setup.

Another note from before running experiments -- I had to humble my grid search, because my original plan was:

    "batch_size": [500, 1000, 2000],
    "lr": [0.1, 0.01, 0.001, 0.0001, 0.00001],
    "optimizer": ["sgd", "adam"],
    "lr_scheduler": ["stepLR", "multiStepLR", "expLR"],
    "random_emb": [True, False],
    "emb_training": [True, False],
    "emb_projection": [True, False],
    "final_metric": ["cos", "neural"],
    "vocab_size": [10000, 20000, 50000]

Which would result in 4320 combinations -- that would be a lot of experiments.

Final grid search looks like this:

    "batch_size": [1000],
    "lr": [0.01, 0.001, 0.0001, 0.00001],
    "optimizer": ["sgd", "adam"],
    "lr_scheduler": ["multiStepLR", "expLR"],
    "random_emb": [True, False],
    "emb_training": [True, False],
    "emb_projection": [True, False],
    "final_metric": ["cos", "neural"],
    "vocab_size": [20000, 50000]

Everything has options count of powers of 2, so the resulting number of combinations is 512.
Which is OK number, since I will be running everything once with the deterministic seed of 9.
Then I will run the best 3 performing combinations 10 times each (with the same seed, different each time of course).

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
