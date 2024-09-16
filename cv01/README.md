# KIV/ANLP assignment 01

## Implement Training Loop and Experiment with Hyper-Parameters

## Prerequisities
1. Instal PyTorch (https://pytorch.org/)
2. Run Hello world (cpu/gpu)
3. Create account on MetaCentrum

## Tasks 

### Our team work  [0pt]

Complete missing parts and design clear interface for experimenting.
1. Use python argparser 
2. Use wandb and log everything
3. For easy login and testing use environment variable WANDB_API_KEY 
4. Run minimalistic hello world on MetaCentrum



### Individual work [4pt]
1. Implement missing fragments in template main01.py
2. Implement 3-layer MLP with ReLU activation function **CF#Dense** 
3. Run Experiments **[2pt]**
   1. Run at least 5 experiments with all possible combinations of following hyper-parameters 
   2. Draw parallel coordinates chart and add image output into output section in this README.md
   

            `model: ["dense", "cnn"]`
            `lr: [0.1, 0.01, 0.001, 0.0001, 0.00001]`
            `optimizer: ["sgd","adam"]`
            `dp: [0, 0.1, 0.3, 0.5]`

   Each experiment train at least for 2 epochs.

 

4. Utilize MetaCentrum **[2pt]**

   For HP search modify attached scripts and utilize cluster MetaCentrum. 
https://metavo.metacentrum.cz/


# My results
## Hyper Parameter Analysis
### Parallel Coordinate Chart
_MISSING_

### Discussion
_MISSING_

# K zamyšlení
1. Podle čeho odhadnout velikost batche?                                                                                          
2. Jaký následek má větší/menší batch? 
3. Jaký má dopad velikost batche na výslednou úspěšnost systému?
4. Jaké má výhody/nevýhody, pokud počítáme test na menším/větším počtu datových vzorků? 
   Kdy byste použili takovou techniku? 
5. Podle čeho nastavit počet epoch při trénování modelů?
6. Proč začíná test a train loss na podobných hodnotách, mohou mít začáteční hodnoty nějaký speciální význam?
7. Je nějaký důvod proč nastavit batch_size pro train/dev/test jinak?
8. Kdy je vhodné použít LR decay?

                                                                   

