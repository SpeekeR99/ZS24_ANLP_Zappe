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

_MISSING_

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

Present mean and std of the dataset. 

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

_MISSING_

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Baseline analysis **[2pt]**
What would the loss of a model returning a random value between 0 and 6 uniformly look like?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

_MISSING_

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

What would the loss of a model returning best prior (most probable output) look like?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

_MISSING_

![#008000](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Implement Dummy Model
1. **Analyze the Dataset**

    #### **CF\#1**

    Count occurrences of words in the datasset, and prepare a list of
    top\_n words

    **CKPT\#1**

2. **Prepare Word Embeddings**.
    https://drive.google.com/file/d/1MTDoyoGRhvLf15yL4NeEbpYLbcBlDZ3c/view?usp=sharing

    [//]: # (# https://fasttext.cc/docs/en/crawl-vectors.html)
    [//]: # (# EMB_FILE = "b:/embeddings/Czech &#40;Web, 2012, 5b tokens &#41;/cztenten12_8-lema-lowercased.vec")

    #### **CF\#2**

    Use the *list of top N words* for pruning the given Word embeddings.
    The cache will be stored on the hard drive for future use **CF\#3**.

    You should see two new files *word2idx.pckl* and *vecs.pckl*
    **CKPT\#2**

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

_MISSING_

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
