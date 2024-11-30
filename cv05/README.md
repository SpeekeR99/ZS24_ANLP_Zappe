# KIV / ANLP Exercise 05

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



![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### sentiment charts

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`



![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Table of results ##

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

|                | Baseline          | Czert-B-base-cased | robeczech-base | FERNET-C5 |
|----------------|-------------------|--------------------|----------------|-----------|
| sts            | -                 | -                  | -              | -         |
| sentiment-csfd | -                 | -                  | -              | -         |

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

We run 5 experiments for each setup and present average and error.
The baseline is taken from previous exercises in the semester. Please write details for the baseline

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

_MISSING_

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Discussion

Který model byl nejlepší? Proč? 

Jaké parametry jsem tunil?

Které měly největší vliv?

Jaké další techniky pro stabilnější nebo lepší výsledky jsem použil?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`



![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`
