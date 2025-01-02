# KIV / ANLP Exercise 05


✅ 20

👍

❌ 


*Maximum points: 20 + 5* 
**Deadline**: 15. 12. 2023 23:59:59

# Implement and run experiments [10pt] 
1. **Run Experiments with different Hyper-parameters [5pt]** 
    
       "task":"sts",     # < 0.65
       #"task": "sentiment",  # > 0.75

        "model_type": "UWB-AIR/Czert-B-base-cased",
        #"model_type": "ufal/robeczech-base",
        #"model_type": "fav-kky/FERNET-C5",

    **The best performing configuration run at least 5 times for each model and dataset** 
    To these runs add special tag : `best`
    Conclude your results in table of results.

2. You can use python transformer package [https://huggingface.co/] [https://huggingface.co/docs/transformers/en/index]:
 - AutoTokenizer.from_pretrained(model_name)
 - AutoModelForSequenceClassification.from_pretrained(model_name)
 - AutoModel.from_pretrained(model_name)


3. **[5pt]** **Tune More Voluntarily**
You can use any technique and model. 
  - If you are interested in experimenting with prompting techniques or other models, please let me know, and we can discuss a way forward. 
 
Best performing student gets 5 extra points. 
If confidence intervals of more students overlay, each student gets extra points. 

# My results **[10pt]** 
## Hyper Parameter Analysis

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I did a rather smaller grid search for this exercise

    model_types=["UWB-AIR/Czert-B-base-cased", "ufal/robeczech-base", "fav-kky/FERNET-C5"]
    tasks=["sts", "sentiment"]
    lrs=[0.001, 0.0001, 0.00001]
    batch_sizes=[32, 64]
    seq_lens=[64, 128]

There are 72 possible combinations total, but when taking model type and task into account it's only 12.

I ran everything about 3 times and selected the best performing configuration for each model and task combination.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Parallel Coordinate Chart

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Here is the result parallel coordinate chart for all the combinations for the STS task (regression -- thus only test_loss)

![HP](img/sts_parallel.svg?raw=true "STS parallel coordinate chart")

Here are the result parallel coordinate charts for all the combinations for the sentiment task (classification -- thus test_loss and test_acc)

![HP](img/sentiment_parallel_loss.svg?raw=true "Sentiment parallel coordinate chart")

![HP](img/sentiment_parallel_acc.svg?raw=true "Sentiment parallel coordinate chart")

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Discussion

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

We can generally see that all the models perform similarly good, but sometimes there are some outliers that performed badly.
That is probably because of the randomness in the training process.

The best performing model for each model and task combination were:

    STS:
    UWB-AIR/Czert-B-base-cased: lr=0.00001, batch_size=64, seq_len=64
    ufal/robeczech-base: lr=0.00001, batch_size=32, seq_len=64
    fav-kky/FERNET-C5: lr=0.00001, batch_size=32, seq_len=64
---
    Sentiment:
    UWB-AIR/Czert-B-base-cased: lr=0.0001, batch_size=64, seq_len=128
    ufal/robeczech-base: lr=0.00001, batch_size=32, seq_len=128
    fav-kky/FERNET-C5: lr=0.0001, batch_size=64, seq_len=128

We can generally say that the lower learning rate (0.0001) was the best for both tasks.
The batch size varies. For the STS task, the best sequence length was always 64; for the sentiment task, it was 128.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Charts
### STS charts

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![HP](img/best_sts_parallel.svg?raw=true "STS parallel coordinate chart")

On the chart above we can see, that the worst performing model for the STS task was the `ufal/robeczech-base` model.

On the other hand, `UWB-AIR/Czert-B-base-cased` and `fav-kky/FERNET-C5` performed similarly good.

On the next chart we can see the training process of the overall best run for the STS task.

![best](img/best_sts_train_loss.svg?raw=true "STS training loss")

![best](img/best_sts_test_loss.svg?raw=true "STS test loss")

The training loss has a perfect decreasing trend, but the test loss is a bit weird.
It started to increased after a few epochs, but then it started to decrease again.
This kind of makes sense, because the model is learning to fit our concrete task, I guess.
 
👍💔 -- I would log with smaller granularity (several batches with large models) -- you could see more interesting progress during the first epoch. Isn't it rather strange that test loss is the best after the first epoch...?

(This is `fav-kky/FERNET-C5` model with lr=0.00001, batch_size=32, seq_len=64)

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### sentiment charts

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

![HP](img/best_sentiment_parallel_loss.svg?raw=true "Sentiment parallel coordinate chart")

![HP](img/best_sentiment_parallel_acc.svg?raw=true "Sentiment parallel coordinate chart")

From the charts above we can see that the `UWB-AIR/Czert-B-base-cased` had the highest loss, `ufal/robeczech-base` had the lowest loss.

We can also see that despite the loss, the `UWB-AIR/Czert-B-base-cased` and `ufal/robeczech-base` models had the lowest accuracy.

Clearly, the `fav-kky/FERNET-C5` model was the best for the sentiment task (according to `accuracy` metrics).

On the next chart we can see the training process of the overall best run for the sentiment task.

![best](img/best_sentiment_train_loss.svg?raw=true "Sentiment training loss")

![best](img/best_sentiment_test_loss.svg?raw=true "Sentiment test loss")

![best](img/best_sentiment_test_acc.svg?raw=true "Sentiment test accuracy")

The training loss has a perfect decreasing trend, but the test loss is a lot weirder (even more weird that for the sts task).
The test accuracy has a perfect increasing trend, which is good.
I honestly have no idea why the test loss behaves this oddly.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Table of results ##

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Random model baseline for the STS task (taken from cv02):

|       | MSE Loss (mean ± std dev) |
|-------|---------------------------|
| Train | 7.784 ± 0.024             |
| Test  | 6.230 ± 0.133             |

Any of the following models trained in the cv03 exercise can be taken as a baseline.
I personally took the `mean` model, because it was supposed to be a baseline in the exercise too:

| MODEL    | TRAIN ACC ± CONFIDENCE INTERVAL | VAL ACC ± CONFIDENCE INTERVAL | TEST ACC ± CONFIDENCE INTERVAL |
|----------|---------------------------------|-------------------------------|--------------------------------|
| **mean** | **0.836 ± 0.024**               | **0.759 ± 0.001**             | ***0.755 ± 0.001***            |
| cnn A    | 0.704 ± 0.030                   | 0.664 ± 0.003                 | *0.657 ± 0.004*                |
| cnn B    | 0.713 ± 0.043                   | 0.704 ± 0.006                 | ***0.700 ± 0.005***            |
| cnn C    | 0.767 ± 0.033                   | 0.723 ± 0.003                 | ***0.705 ± 0.002***            |

Results for each model on each task (`sts` -- `test_loss`; `sentiment` -- `test_acc`):

(used confidence for confidence interval calculation is 95 %)

|                | Baseline      | Czert-B-base-cased | robeczech-base | FERNET-C5         |
|----------------|---------------|--------------------|----------------|-------------------|
| sts            | 6.230 ± 0.133 | 0.513 ± 0.007      | 0.581 ± 0.017  | **0.500 ± 0.008** |
| sentiment-csfd | 0.755 ± 0.001 | 0.843 ± 0.002      | 0.843 ± 0.002  | **0.859 ± 0.003** |

(safe to say that `Random model` baseline for `sts` task is obviously way easier to beat than the `Mean model` baseline for the `sentiment` task)

💔 It would be better to use stronger baseline for STS. This table would look better. 

The best performing model overall was the `FERNET-C5` model; I hate to say it, but that is one thing the `kky` did right.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

We run 5 experiments for each setup and present average and error.
The baseline is taken from previous exercises in the semester. Please write details for the baseline

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

For the STS task, the baseline was a random model, which is easy to beat, but it was also easy to analyze, since the analysis is done in the `cv02` exercise.

For the sentiment task, the baseline was the `mean` model, which was supposed to be a baseline in the `cv03` exercise too.
In the `cv03` exercise though, it was not beaten by any of the convolutional models, so it is a good baseline (hard to beat).
In this exercise all the models beat the baseline easily, which just points out that the models are good.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Discussion

Který model byl nejlepší? Proč? 

Jaké parametry jsem tunil?

Které měly největší vliv?

Jaké další techniky pro stabilnější nebo lepší výsledky jsem použil?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

The overall "best" model was the `FERNET-C5` model, which was the best for both the tasks.

The parameters I tuned were `learning rate`, `batch size`, and `sequence length` (see above).

![importance](img/importance_test_loss.png?raw=true "Importance")

![importance](img/importance_test_acc.png?raw=true "Importance")

Based on the pictures above, the most important parameter to tune was the `seq_len` from the point of view of the `test_loss` metric.
From the point of view of the `test_acc` metric, the most important parameter was the `batch_size` and `lr`.

It is safe to say that the grid search was rather small, so all the parameters were equally important.

I did not use any other techniques for better results, the results were good already (at least good enough for me).

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`
