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
### Parallel Coordinate Chart
_MISSING_
### Discussion
_MISSING_

## Charts
### STS charts (15 charts)
_MISSING_

### sentiment charts (15 charts)
_MISSING_ 

## Table of results ##
|                | Baseline          | Czert-B-base-cased | robeczech-base | FERNET-C5 |
|----------------|-------------------|--------------------|----------------|-----------|
| sts            | -                 | -                  | -              | -         |
| sentiment-csfd | -                 | -                  | -              | -         |

We run 5 experiments for each setup and present average and error.
The baseline is taken from previous exercises in the semester. Please write details for the baseline
_MISSING_


### Discussion
_MISSING_

Který model byl nejlepší? Proč? 

Jaké parametry jsem tunil?

Které měly největší vliv?

Jaké další techniky pro stabilnější nebo lepší výsledky jsem použil?

