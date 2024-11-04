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

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![Histogram](img/training_labels_hist.svg?raw=True "Histogram")

![Histogram](img/testing_labels_hist.svg?raw=True "Histogram")

The dataset looks balanced.

For this task, I have also created histograms of sequence lengths, for later use in the config (`config["seq_len"]`).

![Histogram](img/training_seq_len_hist.svg?raw=True "Histogram")

![Histogram](img/testing_seq_len_hist.svg?raw=True "Histogram")

    Mean training sequence length: 50.554254517156686
    Std dev training sequence length: 51.75246448261309
    Mean testing sequence length: 50.75292701608491
    Std dev testing sequence length: 52.29935225534991
    Q3 training sequence length: 66.0
    Q3 testing sequence length: 66.0
    90-percentile training sequence length: 112.0
    90-percentile testing sequence length: 114.0

| Seq Len Statistics | Train         | Test          |
|--------------------|---------------|---------------|
| Mean ± Std Dev     | 50.55 ± 51.75 | 50.75 ± 52.30 |
| Q3; 75 %           | 66.00         | 66.00         |
| 90-percentile      | 112.00        | 114.00        |

For the reason of the high standard deviation, I have decided to use the mean + std dev as the sequence length -- approximately 100.
Which is somewhere between the 75 % quantile and the 90-percentile, which seems as reasonable sequence length to me.
Extremely long sequences are not that common (100+ words), so we are not loosing that much information and shorter sentences will get noised with padding.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

2. **Prepare Word Embeddings**.
    https://drive.google.com/file/d/1MTDoyoGRhvLf15yL4NeEbpYLbcBlDZ3c/view?usp=sharing

[//]: # (# https://fasttext.cc/docs/en/crawl-vectors.html)
[//]: # (# EMB_FILE = "b:/embeddings/Czech &#40;Web, 2012, 5b tokens &#41;/cztenten12_8-lema-lowercased.vec")
Use the *list of top N word*s for pruning the given Word embeddings.

!! - - IMPORTANT - - !!

**Force vocab size (N most frequent words from the train dataset)
Words without embedding in the given emb file initialize randomly**

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Unit tests in `cv02` literally wanted us to not force vocab size and to prune / not add, if the word is unknown.

Tests now want us to force vocab size and initialize randomly -- for this reason I have changed your tests imports and re-implemented the function `load_ebs`, because I want my tests from `cv02` to pass.

All I have changed in the `test.py` is `from cv02.main02 import load_ebs` -&gt; `from cv03.main03 import load_ebs`.

Please  don't kill me for changing `test.py`; I have not changed anything else.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

The cache will be stored on the hard drive for future use.
You should see two new files *word2idx.pckl* and *vecs.pckl*

3. **Prepare Dataset for Training**
 
Use load_dataset from datasets to load csv dataset.

    from datasets import load_dataset

    cls_dataset = load_dataset("csv", delimiter='\t', data_files={"train": [CSFD_DATASET_TRAIN],
                                                                "test": [CSFD_DATASET_TEST]})
4. **Implement training loop**
   1. Implement basic training loop. 
   2. Implement testing for model and dataset
   

5. **Implement Embedding Averaging Model - Our Baseline** 

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

6. **Implement CNN Model**

![architectures](img/ANLP_cv_03.png)

For implementing architecture of the model use configuration in form of list, 
where each item correspond to a setup of one layer of the model (prepared in **CF\#CNN_CONF**).  

Primitives to use:
- nn.Embedding
- nn.Conv1d, nn.Conv2d
- nn.MaxPool1d, nn.MaxPool2d
- nn.Dropout
- nn.Linear

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I have fiddled with the `hidden_size` parameter and changed the values from

    A = 500
    B = 514
    C = 35000

to

    A = 505
    B = 979
    C = 35000

because the test `test_parameters_number` was failing with the default values provided in the original code.

This might be due to my change of architecture `B`, where I changed the kernels from

    (2, 2)
    (3, 2)
    (4, 2)

to

    (2, reduced_emb_size / 2)
    (3, reduced_emb_size / 2)
    (4, reduced_emb_size / 2)

so that the `B` architecture is right in between the `A` and `C` architectures.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`


7. **Log these:**


    MANDATORY_HP = ["activation", "model", "random_emb", "emb_training", "emb_projection", "lr", "proj_size", "batch_size"]
    MANDATORY_HP_CNN = ["cnn_architecture", "n_kernel", "hidden_size"]
    MANDATORY_M = ["train_acc", "test_loss", "train_loss"] 

8. **Run Experiments with different Hyper-parameters** 

9. **[5pt]** **The best performing experiment run at least 10 times** 
    To these runs add special tag : `best`

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I simply chose the best by grouping the runs by `name` and filtering the `model` and `test_acc`

The "best" mean model seems to be (grouped `test_acc` = 0.759):

    batches=5000
    batch_size=64
    lr=0.001
    activation=relu
    random_emb=False
    emb_training=True
    emb_projection=True

The "best" CNN models seem to be:

Architecture A (grouped `test_acc` = 0.656):

    batches=2000
    batch_size=64
    lr=0.0001
    activation=relu
    random_emb=True
    emb_training=True
    emb_projection=True

Architecture B (grouped `test_acc` = 0.704):

    batches=2000
    batch_size=32
    lr=0.001
    activation=relu
    random_emb=False
    emb_training=True
    emb_projection=True

Architecture C (grouped `test_acc` = 0.704):

    batches: 10000
    batch_size: 32
    lr: 0.0001
    activation: relu
    random_emb: False
    emb_training: True
    emb_projection: True

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

10. **[5pt]** **Tune More Voluntarily**

You can use any technique in scope of CNN architecture. 
Best performing CNN gets 5 extra points. 
If confidence intervals of more students overlay, each student gets extra points.

# My results **[5pt]** 
## Hyper Parameter Analysis
### Parallel Coordinate Chart

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![Parallel Coordinate Chart](img/parallel_coordinate_chart_all.svg?raw=True "Parallel Coordinate Chart")

(note: architecture A seems to be superior, but keep in mind, that the `mean` model might be affecting that, since it had it's "cnn architecture" set to A (for simpler -- unified -- running of the script))

Following charts show the Hyper Parameters better, because they are split by the `model` into two charts:

(mean)

![Parallel Coordinate Chart](img/parallel_coordinate_chart_mean.svg?raw=True "Parallel Coordinate Chart")

(cnn)

![Parallel Coordinate Chart](img/parallel_coordinate_chart_cnn.svg?raw=True "Parallel Coordinate Chart")

The following charts show the Parallel Coordinate Chart for runs that had either over 0.7 `test_acc` or over 0.75 `test_acc` (this is done before running the `best` runs, so the `best` runs):

(test_acc > 0.7)

![Parallel Coordinate Chart](img/parallel_coordinate_chart_over_0_7.svg?raw=True "Parallel Coordinate Chart")

(test_acc > 0.75)

![Parallel Coordinate Chart](img/parallel_coordinate_chart_over_0_75.svg?raw=True "Parallel Coordinate Chart")

From the last one, we can see that the `mean` model outperforms every other model and that the `learning_rate` with the value of 0.001 seems to be the best too.
Also the pretrained embeddings seem to be better than the random embeddings with the ability to train them also.

TODO: parallel coordinate chart pro `best` runs

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Confusion matrix -- best run ##

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

TODO: confusion matrix pro `best` runs

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Discussion

Which hyperparameters did I tune?

Which had the greatest influence?

Have I used other techniques to stabilize the training, and did I get better results?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Side note regarding the `tests`: `test_clss_dist` is failing for me, because my coverage is `0.7969` and the test tests whether the coverage is in the interval `(0.68, 0.78)`.

Should the test be updated and maybe check for coverage `>= 0.68` rather than `== 0.73 ± 0.05`?

---

Important note about my terminology:

`train_loss`, `train_acc` -- these are the values from the training set.

`val_loss`, `val_acc` -- these are the values from the validation set (training set split).

`test_loss`, `test_acc` -- these values are not going to be curves on the plots, because I get to know them and log them only once -- at the end of the training (data the model has never seen).

(I know You mentioned a different terminology, so I believe my `validation` == Your `dev`, but I am not entirely sure)

---

My first version of final grid search looked as follows:

```python
models = ["mean", "cnn"]
batches = [100_000, 500_000]
batch_sizes = [32, 64]
lrs = [0.001, 0.000_1, 0.000_01, 0.000_001]
activations = ["relu", "gelu"]
random_embs = [True, False]
emb_trainings = [True, False]
emb_projections = [True, False]
cnn_architectures = ["A", "B", "C"]
```

Since `mean` model doesn't need to be trained with different `cnn_architecture` values, we basically have 4 total models.
Every other hyperparameter has powers of 2 possibilities, so this makes for a total of 1 024 runs (nice number!).

Anyways, why "first version"? Because the default value of batches was `100 000` and `500 000` (two different values across the files from You).
I once again believed the default values to be correct and ran the first 1024 runs for no reason.
I have later altered the `batches` to be values `10 000` and `20 000`, which still seemed as way too many, so later I reduced it to `2 000` and `5 000`, which was later further decreased to `1 000` and `2 000`.

Side note: why are we training until we hit the amount of batches? Why not use epochs normally? I don't understand this.

---

***BIG PROBLEM***: I have had this problem since the very beginning, but this exercise annoyed me so much I have to write this out.
Wandb server on KIV / Wandb overall is NOT PREPARED for distributed computing.
Most of my jobs on MetaCentrum are failing, because the server cannot handle "Too Many Requests for url: ...".
I tried to run the jobs with `plzen=True` to "virtually limit" the number of jobs, that run at the same time, but it didn't help.
Instead I seem to have run out of mana, because MetaCentrum does not want to run my jobs for 3 days now.
So I am back at the general queue and things are not great.
This problem was not a big deal in the first exercise, but the second one (last one) was a big problem already.
I currently (numbers may change in the following few days, but only a bit) have 4630 runs overall for this exercise, out of which only 1813 finished successfully -- others failed mostly because of Wandb "Too Many Requests" error, or random errors like "Wandb Innit failed", or "raise Exception("problem")" (which is a funny one :) ).

***BIG PROBLEM 2*** (related to above): I am unable to run the `tests` for this exercise, because I am getting `HTTPError: 429 Client Error: Too Many Requests for url: https://api.wandb.ai/graphql` error.
I have been trying to run the `tests` for the last two days (writing this at Sunday night -- so whole weekend `tests` are unusable because of Wandb errors).

Update: Monday morning, `tests` are working now, I believe it is because all my runs have finished on the MetaCentrum and possibly someone else was working on the exercise over the weekend too, so the server is not as overloaded now.

Side note: the tests seem to download all the runs and then check if the tag is there and if everything is correct, wouldn't it be better (if possible) to pre-filter the runs on the server side and only send the runs with the correct tag to the client side?
Or at least maybe be less verbose, because std out makes the tests run real slow.

---

Back to the discussion about the hyperparameters.

![Parameter Importance](img/parameter_importance_test_acc.png?raw=True "Parameter Importance")

![Parameter Importance](img/parameter_importance_test_loss.png?raw=True "Parameter Importance")

As I expected, the biggest importance seems to be the `learning rate` (as usually).
Next the embeddings make the biggest differences too -- `random_emb`, `emb_training`, `emb_projection`.

Suprisingly, the `architecture` of the CNN with the valu `C` seems to be also important, but I don't understand why other values (`A`, `B`) are not in that list at all.

Batches were not really that important for the training itself as for the MetaCentrum runs, because the original values made the jobs not be able to finish in time.

ReLU and GELU activations seem to have similar importance -- not that big of a difference between the two, but all the `best` runs had `ReLU` activation.

Let's do a closer analysis for each of the HPs (I will show everything on the `test_acc`):

1. `model`:
    - Here I have thought, that the CNN's would outperform the mean model, but if I group by `model` the outcome is as follows:
    - ![Model](img/test_acc_groupby_model.svg?raw=True "Model")
    - The `mean` model seems to be outpeforming the `cnn` models by a little bit.
2. `batches`:
    - I cannot really show any good chart here, because none of the `cnn` models were able to finish in time for `batches` = 20 000 nor 10 000.
    - For this reason, the other values are biased, because the `mean` model was able to finish on them, and some good `cnn` models too (mostly `batches` = 2 000 is biased and has the most finished runs).
3. `batch_size`:
    - I personally had no personal preference here, but I have tried batch sizes 32 and 64.
    - ![Batch Size](img/test_acc_groupby_batch_size.svg?raw=True "Batch Size")
    - As we can see, the `batch_size` = 64 seems to be overall better
4. `learning_rate`:
    - I have tried learning rates 0.001, 0.0001, 0.00001, 0.000001. and I personally expected the 0.001 (highest one) to be the best.
    - ![Learning Rate](img/test_acc_groupby_lr.svg?raw=True "Learning Rate")
    - The `learning_rate` = 0.0001 seems to be the best, but my choice (0.001) is not that far behind.
5. `activation`:
    - I have tried `ReLU` and `GELU` activations.
    - Since both are very similar, I had no preference here.
    - ![Activation](img/test_acc_groupby_activation.svg?raw=True "Activation")
    - The `ReLU` activation seems to be slightly better over the `GELU` activation.
6. `random_emb`:
    - As before, I expected the random embedding initialization to be worse than the pretrained embeddings.
    - ![Random Emb](img/test_acc_groupby_random_emb.svg?raw=True "Random Emb")
    - To my surprise, they both seem to be very similar (again, as last exercise).
    - What is more, the random initialization seems to be very slightly better.
7. `emb_training`:
    - As before, I expected the embeddings to be better when trained.
    - ![Emb Training](img/test_acc_groupby_emb_training.svg?raw=True "Emb Training")
    - Truly, the trained embeddings seem to be way better, which is no surprise.
    - It makes sense, because the embeddings are trained on the same data, so they should be better.
8. `emb_projection`:
    - As before, I expected the projection of embeddings to be worth it and be better.
    - ![Emb Projection](img/test_acc_groupby_emb_projection.svg?raw=True "Emb Projection")
    - And it is, the projection of embeddings seems to be better off.
9. `cnn_architecture`:
    - Here, it is interesting, because my architecture `A` has kernel sizes `(x, 1)`, architecture `B` has kernel sizes `(x, reduced_emb_size / 2)`, and architecture `C` has kernel sizes `(x, reduced_emb_size)`.
    - So it is the matter of the kernel sizes and I personally the middle ground -- `B` -- to be the best. Simply because `A` pays too much attention to details -- the words themselves, and `C` pays almost no attention to detail -- pays attention to the whole sentences (?).
    - My graphs here are not that good again, because `A` is biased by the `mean` model, which was set to `A` for the sake of unified running of the script.
    - But from filtering and grouping and looking by eye, the `B` architecture seems to be the best (on average), but it is closely followed by the `C` architecture.

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
