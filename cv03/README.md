# KIV / ANLP Exercise 03

*Deadline to push results:* 2024-11-10 (San) 23:59:59

*Maximum points:* 

The Goal
========

Implement missing parts in the given template of the supervised machine
learning stack for predicting sentiment of given sequence (Sentiment Classification Task). 
Train and evaluate the system on the given data set. Use architectures described in the
following section.

What is Semantic Sentiment Analysis 
===================================
Varies by domain (most common: 2-3 classification) It is not a dogma...

CSFD Sentiment Czech Dataset (user ranking 0-100%): positive,neutral,negative

Project Structure 
=================

- [tests]  
- [data]
    -   *csfd-train.tsv*
    -   *csfd-test.tsv*
- *main03.py*


The Data Set
============

- Your dataset is split : train/test
- Your dataset has labels in tsv 
  - negative:0 
  - neutral:1 
  - positive:2


Tasks \[20+5 points in total\]
===============================

1. **Analyze the Dataset**

**CF\#STATISTICS**

- You can use scripts from Exercise 02 as a starting point.
- Count occurrences of words in the data set, and prepare a list of
top\_n words
- Count statistics about coverage of tokens in training dataset 
- Coverage is ratio between tokens you have in your vocabulary and all tokens.
Do not count pad tokens
- Count statistics about class distribution in dataset (train/test)

2. **Prepare Word Embeddings**.
    https://drive.google.com/file/d/1MTDoyoGRhvLf15yL4NeEbpYLbcBlDZ3c/view?usp=sharing

[//]: # (# https://fasttext.cc/docs/en/crawl-vectors.html)
[//]: # (# EMB_FILE = "b:/embeddings/Czech &#40;Web, 2012, 5b tokens &#41;/cztenten12_8-lema-lowercased.vec")
Use the *list of top N word*s for pruning the given Word embeddings.

!! - - IMPORTANT - - !!

**Force vocab size (N most frequent words from the train dataset)
Words without embedding in the given emb file initialize randomly**


The cache will be stored on the hard drive for future use.
You should see two new files *word2idx.pckl* and *vecs.pckl*


3. **Prepare Dataset for Training**

 
Use load_dataset from datasets to load csv dataset.

    from datasets import load_dataset

    cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [CSFD_DATASET_TRAIN],
                                                                "test": [CSFD_DATASET_TEST]})
7. **Implement training loop**
   1. Implement basic training loop. 
   2. Implement testing for model and dataset
   

8. **Implement Embedding Averaging Model - Our Baseline** 

Implement model which uses average of sequence embeddings to represent the sequence.

The Model takes sequence of numbers as an input [SEQ_LEN]. Use prepared word embeddings (task-2) to lookup word
vectors from ids [SEQ_LEN,EMB_SIZE]. Freeze embeddings. Add one trainable projection layer on top of the
embedding layer [SEQ_LEN,EMB_PROJ]. Use the mean of all words vectors in a sequence as
a representation of the sequence [EMB_PROJ].

Add classification head [NUM_OF_CLASSES].

Primitives to use:
- nn.Embedding 
- nn.Softmax
- nn.Linear
- nn.Dropout
- nn.[activation]

**[5pt]**

10. **Implement CNN Model**
![architectures](img/ANLP_cv_03.png)
For implementing architecture of the model use configuration in form of list, 
where each item correspond to a setup of one layer of the model (prepared in **CF\#CNN_CONF**).  

Primitives to use:
- nn.Embedding
- nn.Conv1d, nn.Conv2d
- nn.MaxPool1d, nn.MaxPool2d
- nn.Dropout
- nn.Linear
    

11. **Log these:**


    MANDATORY_HP = ["activation", "model", "random_emb", "emb_training", "emb_projection", "lr", "proj_size", "batch_size"]
    MANDATORY_HP_CNN = ["cnn_architecture", "n_kernel", "hidden_size"]
    MANDATORY_M = ["train_acc", "test_loss", "train_loss"] 

13. **Run Experiments with different Hyper-parameters** 

13. **[5pt]** **The best performing experiment run at least 10 times** 
    To these runs add special tag : `best`

14. **[5pt]** **Tune More Voluntarily**
You can use any technique in scope of CNN architecture. 
Best performing CNN gets 5 extra points. 
If confidence intervals of more students overlay, each student gets extra points. 


# My results **[5pt]** 
## Hyper Parameter Analysis
### Parallel Coordinate Chart
_MISSING_

## Confusion matrix -- best run ##
__MISSING__

### Discussion
_MISSING_

Which hyperparameters did I tune?
Which had the greatest influence?
Have I used other techniques to stabilize the training, and did I get better results?



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

                                            











