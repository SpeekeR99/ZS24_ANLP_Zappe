# CV04 - Named Entity Recognition/Morphological Tagging

**Deadline**: 20. 11. 2023 23:59:59

**Maximum points:** 20 + 5 (bonus)

**Contact:** Jan Pašek (pasekj@ntis.zcu.cz) - get in touch with me in case of any problems, but try not to leave
the problems for the weekends (I can't promise I'll respond during the weekend). Feel free to drop me an email or come to the next lesson
to discuss your problems. I'm also open to schedule online session in the meantime if you need
any support.

## Problem statement:

In this assignment you'll practice the RNN/LSTM-based neural networks for token classification
on two different tasks. The first task will be NER (Named Entity Recognition) and the second will be Morphological Tagging.

**What is NER:** Named entity recognition (NER) is the task of tagging entities in text with 
their corresponding type. Approaches typically use BIO notation, which differentiates the 
beginning (B) and the inside (I) of entities. O is used for non-entity tokens.
(source: https://paperswithcode.com/task/named-entity-recognition-ner)

**What is Tagging:** Morphological tagging is the task of assigning labels to a sequence of 
tokens that describe them morphologically. As compared to Part-of-speech tagging, 
morphological tagging also considers morphological features, such as case, #
gender or the tense of verbs. (source: https://paperswithcode.com/task/morphological-tagging).

To do so, we will use two datasets that are already pre-processed and ready to use
(including the data input pipeline). The first dataset is the [CNEC](https://ufal.mff.cuni.cz/cnec) (Czech Named Entity Corpus)
and is designated for the NER task. The second utilized dataset is the [UD](https://universaldependencies.org) (Universal Dependencies) -
Czech treebanks only. Both corpora are pre-processed to have the same format using labels in [BIO](https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)) notation.

*In addition to the RNN and LSTM models, you will have a change to run some experiments with
BERT-like models (CZERT and Slavic) that are pre-trained for Czech and Slavic languages, respectively.
Thanks to that you will be able to understand the strength of the large pre-trained models.*

## Project Structure
- [data] - data for NER task (do not touch)
- [data-mt] - data for Tagging task (do not touch)
- [test] - unittest to verify your solutions (do not touch)
- main04.py - main source code of the assignment, training loops, etc.
- models.py - implementation of all models
- ner_utils.py - data input pipeline, etc. (do not touch)
- README.md

## Tasks:

### CKPT1 (Dataset Analysis)
Analyse the dataset - write the results into the discussion (secion 1). Answer all the following questions:
1. What labels are used by both datasets - write a complete list and explanation of the labels (use the referenced dataset websited).
2. How large are the two datasets (train, eval, test, overall).
3. What is the average length of a training example for the individual datasets - in number of whole words tokens as pre-tokenized in the dataset files.
4. What is the average length of a token for the individual datasets - in number of subword tokens when using `tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")` - documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer (methods: encode or batch_encode).
5. Count statistics about class distribution in dataset (train/dev/test) for the individual datasets.
6. Based on the statistic from the questions above - are the individual datasets balanced or unbalanced? In case at least one of the dataset is unbalanced, are there any implications for the model/solution or is there anything we should be aware of?

**[3 pt]** - Evaluation method: manually (each question 0.5pt)

### CKPT2 (RNN Model)
*Note: During the whole implementation, preserve the attribute/variable names if suggested (e.g. `self.__dropout_layer = ...` -> `self.__dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)`. If you change the name of such variables, the tests will be failing.*

*Note: When implementing the models, mind the constructor params. All the parameters shall be employed by the model and the model shall adapt it's functionality based on the provided arguments (some parameters may be used in later steps as well -> no panic if you don't use them right now).*

1. State the equations used for computing the activations of an RNN model in the discussion (section 2).
2. Implement an RNN model with the allowed torch primitives only:
  - Allowed:
    - torch.nn.Embedding
    - torch.nn.Dropout
    - torch.nn.Linear
    - torch.nn.CrossEntropyLoss
    - all torch functions (e.g. tensor.view(), torch.tanh(), torch.nn.functional.Softmax(), etc...)
  - Not allowed:
    - torch.nn.Rnn
    - torch.nn.GRU
  - Architecture Description:
    - Inputs (come into the model tokenized using subword tokenizer) are embedded using an embedding layer.
    - Dropouts applied on the embedded sequence.
    - Sequence of hidden states is computed sequentially in a loop using one `torch.Linear` layer with `torch.tanh` activation (you have to save all the hidden states for later)
      - Hint: make sure you make a deep copy of the hidden state tensor preserving the gradient flow (`tensor.clone()`)
    - Dropout is applied to the sequence of hidden states
    - Compute output activations
    - Compute loss and return instance of `TokenClassifierOutput` (*Note: the loss is computed in the forward pass and returned in the `TokenClassifierOutput` to unify interface of our custom models with HuggingFace models*.)
3. Do a step by step debugging to ensure that the implementation works as expected - check the dimensionality of tensors flowing through the model (no points for that, but it is important for the experiments that the model works correctly)

**[4 pt]** - Evaluation method: passing unittests for ckpt2 (3.5pt), discussion manually (0.5pt)

### CKPT3 (LSTM Model)
*Note: During the whole implementation, preserve the attribute/variable names if suggested (e.g. `self.__dropout_layer = ...` -> `self.__dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)`. If you change the name of such variables, the tests will be failing.*

*Note: When implementing the models, mind the constructor params. All the parameters shall be employed by the model and the model shall adapt it's functionality based on the provided arguments (some parameters may be used in later steps as well -> no panic if you don't use them right now).*

1. State the equations used for computing the activations of an LSTM model in the discussion and explain the individual gates (their purpose) (section 3).
2. Implement an LSTM model with any possible primitives :
  - Suggested:
    - torch.nn.Embedding
    - torch.nn.Dropout
    - torch.nn.Linear
    - torch.nn.LSTM
    - torch.nn.CrossEntropyLoss
    - all torch functions (e.g. tensor.view(), torch.tanh(), torch.nn.functional.Softmax(), etc...)
  - Architecture Description:
    - Inputs (come into the model tokenized using subword tokenizer) are embedded using an embedding layer.
    - Dropouts applied on the embedded sequence.
    - **Bi**LSTM layer with parameterizable number of layers is used to process the embedded sequence.
    - Dropout is applied to the sequential output of the LSTM layers
    - A dense layer with ReLu activation is applied
    - A classification head with softmax activation is applied to compute output activations
    - Compute loss and return instance of `TokenClassifierOutput` (*Note: the loss is computed in the forward pass and returned in the `TokenClassifierOutput` to unify interface of our custom models with HuggingFace models*.)
3. Do a step by step debugging to ensure that the implementation works as expected - check the dimensionality of tensors flowing through the model (no points for that, but it is important for the experiments that the model works correctly)

**[3 pt]** - Evaluation method: passing unittests for ckpt3 (2.5pt), discussion manually (0.5pt)


### CKPT4 (Freezing Parameters & L2 Regularization)

1. Implement a possibility to freeze an embedding layer of the RNN and LSTM model - it means that the embedding layer (that we alway init randomly) will not be trained at all.
    - method `self.__freeze_embedding_layer()` - for both models
2. Implement the following methods:
   - `compute_l2_norm_matching` - compute an L2 norm of all model parameters matching a pattern from a given list of patterns (python built-it function `any()` can be useful)
3. Implement `compute_l2_norm` method for both RNN and LSTM model and return the L2 scaled with the `self.__l2_alpha`. Use the previously implemented method
    - RNN: regularize only the dense layer for computing the new hidden state and the classification head
    - LSTM: regularize only the dense layer and classification head
4. In the discussion (section 4) explain in which case do we want to freeze the embedding layer. Also discuss whether it is useful to freeze embedding layer in our case when we initialize the embedding layer randomly - would you expect the model to work well with the frozen randomly initialized embedding layer?

**[3 pt]** - Evaluation method: passing unittest for ckpt4 (2pt), discussion manually (1pt)

### CKPT5 (Training loop & LR schedule)

*Note: During the whole implementation, preserve the attribute/variable names if suggested (e.g. `self.__dropout_layer = ...` -> `self.__dropout_layer = torch.nn.Dropout(p=self.__dropout_prob)`. If you change the name of such variables, the tests will be failing.*


1. Read through the training loop and understand the implementation. Check the usage of `scheduler` variable. (not evaluated by us, but helpful for you)
2. Implement `lr_schedule()` - LR scheduler with linear warmup during first `warmup_steps` training steps and linear decay to zero over the whole training.
    - The scheduler shall return number in [0, 1] - resulting LR used by the model is `training_args.learning_rate * lr_schedule()`
3. In the discussion (section 5) discuss why such LR scheduler can help to improve results. Discuss both the warmup and decay separately.

**[2 pt]** Evaluation method: passing unittests for ckpt5 (1pt), discussion manually (1pt)

### CKPT6 (Basic Experiments)

1. NER experiments with RNN and LSTM
   - Some hyperparameters to start with:
     - RNN
       - ```shell
          main04.py \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --model_type RNN \
            --data_dir data \
            --labels data/labels.txt \
            --output_dir /scratch.ssd/pasekj/job_3984202.cerit-pbs.cerit-sc.cz/output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 200 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 1000 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 64 \
            --num_lstm_layers 2 \
            --embedding_dimension 128 \
            --task NER
         ```
     - LSTM
       - ```shell
         main04.py \
            --model_type LSTM \
            --data_dir data \
            --labels data/labels.txt \
            --output_dir output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 200 \
            --eval_dataset_batches 200 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 1000 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 128 \
            --num_lstm_layers 2 \
            --embedding_dimension 128
         ```
   - Run at least one experiment with the following hyperparameters changed:
       - use `--no_bias`/don't use `--no_bias`
       - LR: {0.0001, 0.001} (`--learning_rate`)
       - L2 alpha: {0.01, 0} (`--l2_alpha`)
       - It means that you will have at least 6 runs for each model - always use the base hyperparameters and then change just the one
2. TAGGING experiments with RNN and LSTM
   - Some hyperparameters to start with:
     - RNN
       - ```shell
          main04.py \
            --model_type RNN \
            --data_dir data-mt \
            --labels data-mt/labels.txt \
            --output_dir output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 300 \
            --eval_dataset_batches 200 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 10 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 128 \
            --num_lstm_layers 2 \
            --embedding_dimension 128 \
            --task TAGGING
         ```
     - LSTM
       - ```shell
          main04.py \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 8 \
            --model_type LSTM \
            --data_dir data-mt \
            --labels data-mt/labels.txt \
            --output_dir output \
            --do_predict \
            --do_train \
            --do_eval \
            --eval_steps 300 \
            --logging_steps 50 \
            --learning_rate 0.0001 \
            --warmup_steps 4000 \
            --num_train_epochs 10 \
            --no_bias \
            --dropout_probs 0.05 \
            --l2_alpha 0.01 \
            --lstm_hidden_dimension 128 \
            --num_lstm_layers 2 \
            --embedding_dimension 128 \
            --task TAGGING
         ```
   - Run at least one experiment with the following hyperparameters changed:
       - use `--no_bias`/don't use `--no_bias`
       - LR: {0.0001, 0.001} (`--learning_rate`)
       - L2 alpha: {0.01, 0} (`--l2_alpha`)
       - It means that you will have at least 6 runs for each model - always use the base hyperparameters and then change just the one
3. CZERT and Slavic 
    - One experiment with each model for each task (4 runs in total) - hyperparameters provided below:
      - NER
        - CZERT
          - ```shell
              main04.py \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --model_type CZERT \
                --data_dir data \
                --labels data/labels.txt \
                --output_dir output \
                --do_predict \
                --do_train \
                --do_eval \
                --eval_steps 100 \
                --logging_steps 50 \
                --learning_rate 0.0001 \
                --warmup_steps 4000 \
                --num_train_epochs 50 \
                --dropout_probs 0.05 \
                --l2_alpha 0.01 \
                --task NER
            ```
        - SLAVIC
          - ```shell
              main04.py \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --model_type SLAVIC \
                --data_dir data \
                --labels data/labels.txt \
                --output_dir output \
                --do_predict \
                --do_train \
                --do_eval \
                --eval_steps 100 \
                --logging_steps 50 \
                --learning_rate 0.0001 \
                --warmup_steps 4000 \
                --num_train_epochs 50 \
                --dropout_probs 0.05 \
                --l2_alpha 0.01 \
                --task NER
            ```
      - TAGGING
        - CZERT
          - ```shell
             main04.py \
              --per_device_train_batch_size 32 \
              --per_device_eval_batch_size 32 \
              --model_type CZERT \
              --data_dir data-mt \
              --labels data-mt/labels.txt \
              --output_dir output \
              --do_predict \
              --do_train \
              --do_eval \
              --eval_steps 300 \
              --logging_steps 50 \
              --learning_rate 0.0001 \
              --warmup_steps 4000 \
              --num_train_epochs 10 \
              --dropout_probs 0.05 \
              --l2_alpha 0.01 \
              --task TAGGING
            ```
        - SLAVIC
          - ```shell
            main04.py \
              --per_device_train_batch_size 32 \
              --per_device_eval_batch_size 32 \
              --model_type SLAVIC \
              --data_dir data-mt \
              --labels data-mt/labels.txt \
              --output_dir output \
              --do_predict \
              --do_train \
              --do_eval \
              --eval_steps 300 \
              --logging_steps 50 \
              --learning_rate 0.0001 \
              --warmup_steps 4000 \
              --num_train_epochs 10 \
              --dropout_probs 0.05 \
              --l2_alpha 0.01 \
              --task TAGGING
            ```
4. Discussion for NER - compare results of the individual models and try to explain why the models achieve the results you observed. Specifically compare the results achieved with RNN/LSTM and CZERT/Slavic.
5. Discussion for TAGGING - compare results of the individual models and try to explain why the models achieve the results you observed. Specifically compare the results achieved with RNN/LSTM and CZERT/Slavic.

**[5 pt]** Evaluation method: passing unittests for ckpt6 (3pt), discussion manually (2pt) - it is not possible to get any points for the passing unittests if discussion is missing at this CKPT

### CKPT7 (Extended experiments)

1. Use the same hyperparameters for CZERT model as above and make additional experiments with the following hyperparameters on NER:
    - Freeze embedding layer and train the model - do you observe any difference in the achieved results?
    - Freeze first {2, 4, 6} layers of the CZERT model + freeze embeddings - do you observe any difference in the achieved results?
2. Adjust `main04.py` to enable to train another model `BERT` - simple change after line `243` and you can use the Czert model implementation and just provide a different model name. For this experiment, use the `bert-base-cased` (https://huggingface.co/bert-base-cased?text=Paris+is+the+%5BMASK%5D+of+France.). 
   - If you choose to implement this bonus, please use `--model_type BERT` so that unittest can recognize that correctly.
   - This experiment is a test of how well does a pre-trained model for English perform on a Czech tasks.
   - Run at least 5 experiments with different hyperparameters for each task (use MetaCentrum for that). Make sure you've run enough epochs to enable model to converge.
3. Discuss the results of both subtasks and also answer the following questions:
    - Does the model with frozen embeddings perform worse than the model with trainable embeddings?
    - Do you see any result improvement/impairment when freezing the lower layers of the CZERT model?
    - Does freezing the lower layers bring any benefits in term of results, speed of training, etc?
    - Does the BERT model work for Czech tasks? State the results and include a graph of eval metrics for the BERT model config for both tasks.

**[5pt]** Evaluation method: passing unittests for ckpt7 (3pt), discussion manually (2pt)

## Discussions

### Section 1 - Dataset Analysis

1. What labels are used by both datasets - write a complete list and explanation of the labels (use the referenced dataset websited).

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

CNEC (Czech Named Entity Corpus)

    O           Outside         (of any entity)
    I-T         Inside          Time expression
    I-P         Inside          Personal name
    I-O         Inside          Artifact name
    I-M         Inside          Media name
    I-I         Inside          Institution name
    I-G         Inside          Geographical name
    I-A         Inside          Numbers in adress
    B-T         Beginning       Time expression
    B-P         Beginning       Personal name
    B-O         Beginning       Artifact name
    B-M         Beginning       Media name
    B-I         Beginning       Institution name
    B-G         Beginning       Geographical name
    B-A         Beginning       Numbers in adress

---

UD (Universal Dependencies)

    ADJ         Adjective
    ADP         Adposition
    ADV         Adverb
    AUX         Auxiliary verb
    CCONJ       Coordinating conjunction
    DET         Determiner
    INTJ        Interjection
    NOUN        Noun
    NUM         Numeral
    PART        Particle
    PRON        Pronoun
    PROPN       Proper noun
    PUNCT       Punctuation
    SCONJ       Subordinating conjunction
    SYM         Symbol
    VERB        Verb
    X           Other
    _           I looked into train.txt and only "_" and "aby" are labeled with this label
                (I assume it means "Unknown" or "Not labeled" or something similar)

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

2. How large are the two datasets (train, eval, test, overall).

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

CNEC:

	train:      4688 sentences
	dev:        577 sentences
	test:       585 sentences
	Overall:    5850 sentences

UD:

	train:      103143 sentences
	dev:        11326 sentences
	test:       12216 sentences
	Overall:    126685 sentences

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

3. What is the average length of a training example for the individual datasets - in number of whole words tokens as pre-tokenized in the dataset files.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

CNEC:

	train:      25.529223549488055 words
	dev:        25.642980935875215 words
	test:       25.745299145299146 words
	Overall:    25.562051282051282 words

UD:

	train:      17.582337143577362 words
	dev:        16.988698569662724 words
	test:       16.894482645710543 words
	Overall:    17.46293562773809 words


![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

4. What is the average length of a token for the individual datasets - in number of subword tokens when using `tokenizer = transformers.BertTokenizerFast.from_pretrained("UWB-AIR/Czert-B-base-cased")` - documentation: https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer (methods: encode or batch_encode).

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

CNEC:

	train:      35.535196245733786 tokens
	dev:        35.76083188908146 tokens
	test:       36.04957264957265 tokens
	Overall:    35.60888888888889 tokens

UD:

	train:      23.36788730209515 tokens
	dev:        22.416651951262583 tokens
	test:       22.296823837590047 tokens
	Overall:    23.179563484232546 tokens

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

5. Count statistics about class distribution in dataset (train/dev/test) for the individual datasets.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

CNEC:

![Train](img/CNEC_train_label_dist.svg?raw=True "Histogram")

![Dev](img/CNEC_dev_label_dist.svg?raw=True "Histogram")

![Test](img/CNEC_test_label_dist.svg?raw=True "Histogram")

UD:

![Train](img/UD_train_label_dist.svg?raw=True "Histogram")

![Dev](img/UD_dev_label_dist.svg?raw=True "Histogram")

![Test](img/UD_test_label_dist.svg?raw=True "Histogram")

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

6. Based on the statistic from the questions above - are the individual datasets balanced or unbalanced? In case at least one of the dataset is unbalanced, are there any implications for the model/solution or is there anything we should be aware of?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

CNEC:

By definition of the problem the dataset MUST be unbalanced, because most of the things are "Outside of any entity" -- Label "O".)
The dataset is unbalanced, but it should be expected.
The model should be able to handle the unbalanced dataset -- it will probably resolve some lower weights for the common label of "Outside of any entity".
I personally think that the model will be able to achieve good results.

UD:

Again, by definition of the problem, we cannot expect the natural languages to have uniformly distributed parts of speech.
We can safely assume, that nouns will be the most common part of speech, at least for Czech language.
So of course, the dataset is unbalanced, because nouns are the most common part of speech, whereas there is little particles for example.
I personally once again believe, that the model should handle the unbalanced dataset well and that it will achieve good results.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Section 2 - RNN Model

1. State the equations used for computing the activations of an RNN model.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

The next state of the RNN is dependent on the previous hidden state and the current input.

The equations are as follows:

    h_t = σ_h(U * x_t + V * h_(t-1) + b_h)
    y_t = σ_y(W * h_t + b_y)

Where:

    h_t         current hidden state
    σ_h         activation function for the hidden state
    U           weight matrix for the input
    x_t         current input
    V           weight matrix for the previous hidden state
    h_(t-1)     previous hidden state
    b_h         bias for the hidden state
    
    y_t         current output
    σ_y         activation function for the output
    W           weight matrix for the hidden state
    h_t         current hidden state
    b_y         bias for the output

It can be better seen from the following diagram:

![RNN](img/RNN_diagram.png?raw=True "RNN")

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Section 3 - LSTM Model

1. State the equations used for computing the activations of an LSTM model in the discussion and explain the individual gates (their purpose).

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

The LSTM model has 3 gates: forget, input, and output gate.
The gates are used to control the flow of information through the cell state.

- The forget gate decides what information to forget from the previous cell state.
- The input gate decides what information to add to the cell state from the "current input" (candidate cell state).
- The output gate decides what information to output from the cell state to the hidden state.

The equations are as follows:

    f_t = σ_f(W_f * x_t + U_f * h_(t-1) + b_f)
    i_t = σ_i(W_i * x_t + U_i * h_(t-1) + b_i)
    o_t = σ_o(W_o * x_t + U_o * h_(t-1) + b_o)
    c^~_t = σ_c(W_c * x_t + U_c * h_(t-1) + b_c)
    c_t = f_t ○ c_(t-1) + i_t ○ c^~_t
    h_t = o_t ○ σ_h(c_t)

Where:

    ○           element-wise multiplication

    f_t         forget gate
    i_t         input gate
    o_t         output gate
    c^~_t       candidate cell state
    σ_XYZ       activation function for the "XYZ" gate
    W_XYZ       weight matrix for the input and the "XYZ" gate
    U_XYZ       weight matrix for the previous hidden state and the "XYZ" gate
    x_t         current input
    h_(t-1)     previous hidden state
    b_XYZ       bias for the "XYZ" gate

    c_t         current cell state
    c_(t-1)     previous cell state
    h_t         current hidden state
    σ_h         activation function for the hidden state

It can be better seen from the following diagram:

![LSTM](img/LSTM_diagram.png?raw=True "LSTM")

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Section 4 - Parameter Freezing & L2 Regularization

1. Explain in which case do we want to freeze the embedding layer. Also discuss whether it is useful to freeze embedding layer in our case when we initialize the embedding layer randomly - would you expect the model to work well with the frozen randomly initialized embedding layer?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

In my opinion it is useful to freeze the embedding layer when we have a pre-trained embeddings and we do not want to fine-tune them.
We may not want to fine-tune the embeddings, if we for example have a small dataset and we want the general relationships between words in the original embeddings to be preserved.
Our small dataset may cause some previously learned relationships to be lost, if we fine-tuned the embeddings.

In our case, where we initialize the embeddings randomly, it is not good to freeze the embedding layer in my opinion.
Simply because they are random and the model should learn the embeddings during the training process.
It may not be totally fatal for the model, but I expect it to not work well with the frozen randomly initialized embedding layer.
I think it may not be fatal simply because the architecture of the model is not as simple and it might be able to learn some useful information elsewhere that would help it to perform well.
But I still think it is not a good idea to freeze the embeddings in our case.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Section 5 - LR Schedule

1. Discuss why such LR scheduler can help to improve results. Discuss both the warmup and decay separately.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Learning rate scheduler with linear warmup can help to improve results, because we start with a random position in the parameter space.
We want to move slowly at the beginning, because we do not know where the minimum is.
As we start moving towards some local minima, we can increase the learning rate to speed up the process.
Moving on with the training, we want to decrease the learning rate to make sure we do not overshoot the minimum.
Closing to the end of the training, we want to decrease the learning rate even more to make sure we do not oscillate around the minimum.

TL;DR:

    Warmup helps to prevent the model from getting stuck in some local minima.
    Decay helps to prevent the model from overshooting the minimum.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Section 6 - Basic Experiments Results

1. Discussion for NER - compare results of the individual models and try to explain why the models achieve the results you observed. Specifically compare the results achieved with RNN/LSTM and CZERT/Slavic.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Since we figured out that the dataset are (and have to be -- by the definition of the problem) unbalanced, I am going to be projecting the "goodness" of the model based on the F1 score.
That is because the F1 score is a harmonic mean of precision and recall and that makes up for the unbalanced dataset (it is a good metric here in my opinion).

Before answering the main question, I would like to show parallel coordinate charts of the tuned hyperparameters for both tasks.
Tuned hyperparameters are:

    - no_bias (true/false)
    - learning_rate (0.0001/0.001)
    - l2_alpha (0.01/0)

(questionably, NER/TAGGING task can be also taken as a "hyperparameter" of sorts)

![HP](img/Parallel_no_bias.svg?raw=True "Parallel coordinate chart of tuned hyperparameters")

![HP](img/Parallel_lr.svg?raw=True "Parallel coordinate chart of tuned hyperparameters")

![HP](img/Parallel_l2_alpha.svg?raw=True "Parallel coordinate chart of tuned hyperparameters")

From those charts, we cannot see much, both options of each hyperparameter had good runs and bad runs.

![HP](img/Parallel_task.svg?raw=True "Parallel coordinate chart of tuned hyperparameters")

From the last chart though, we can clearly see, that the TAGGING task was overall easier for the models to solve, than the NER task.

Now let's get back to the main question.

I would expect the LSTM model to perform better than the RNN model (if comparing that way).
If we are comparing RNN/LSTM with CZERT/SLAVIC, I would expect the CZERT/SLAVIC to perform better, because they are big pre-trained models.

![MODEL_TYPE_F1](img/NER_model_type_f1_parallel.svg?raw=True "F1 scores for different models")

The graph above shows the mapping of average model (group-by model type) F1 scores for the NER task.

As expected, CZERT and SLAVIC models outperform the RNN and LSTM models.
As expected, the LSTM model outperforms the RNN model.

My runs were run with the default hyperparameters, so the CZERT and SLAVIC models were trained on way less epochs, than RNN and LSTM models, so the following graphs look somewhat weird and bad.

![MODEL_TYPE_LOSS](img/NER_model_type_eval_loss.svg?raw=True "Losses for different models")

As we can see, LSTM started to get overfit, but it still outperformed the RNN model.

![MODEL_TYPE_F1](img/NER_model_type_f1.svg?raw=True "F1 scores for different models")

From the F1 scores, we can see, that the pre-trained models were good right from the start,
whereas the RNN/LSTM models needed some time to get better, but they were not able to catch up
with the pre-trained models even though, they were trained way longer, than the CZERT/SLAVIC models.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

2. Discussion for TAGGING - compare results of the individual models and try to explain why the models achieve the results you observed. Specifically compare the results achieved with RNN/LSTM and CZERT/Slavic.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

As above, I will be using F1 score again, for the same reasons.

Again, I would expect the LSTM model to perform better than the RNN model, simply because it is a more complex model based on RNN,
so it is logical to me, that it should perform better (otherwise why would it exist? -- or ok, it would exist, but nobody would use it).
If we are comparing RNN/LSTM with CZERT/SLAVIC, I would once again of course expect the CZERT/SLAVIC to perform better.

![MODEL_TYPE_F1](img/TAGGING_model_type_f1_parallel.svg?raw=True "F1 scores for different models")

The graph above shows the mapping of average model (group-by model type) F1 scores for the TAGGING task.

As expected, CZERT and SLAVIC models outperform the RNN and LSTM models.
But this time, the LSTM model performed way better than on the NER task, it almost reached the CZERT/SLAVIC models.
RNN model once again, the simplest, the worst.

Here the problem with different epochs is not present, because the dataset was really big and all the models had the same amount of epochs.

![MODEL_TYPE_LOSS](img/TAGGING_model_type_eval_loss.svg?raw=True "Losses for different models")

From the eval_loss graph, we can see, that the LSTM model did not overfit this time.
We can once again see how good the pre-trained big models are from the get-go.

![MODEL_TYPE_F1](img/TAGGING_model_type_f1.svg?raw=True "F1 scores for different models")

On the F1 scores graph, we can see, that the pre-trained big models are constantly good, but we can focus
on the LSTM model and see, that it is getting better and better, by the end of the training it was almost
as good as CZERT or SLAVIC (same for eval_loss graph above).

This surprises me a bit, because I would not expect LSTM to be that good, but it is really good.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

### Section 7 - Extended Experiments Results (Bonus)

1. Does the model with frozen embeddings perform worse than the model with trainable embeddings?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

Since we are comparing CZERT and BERT model, I would always expect CZERT to be better here, because it was trained on Czech data.
Bert was trained on English data, so it should not be bad, but it should not be as good as CZERT, especially when freezing the embeddings.

I would expect CZERT with trainable embeddings to be the best, because it can learn the embeddings during the training process.
On the other hand, I would expect BERT with frozen embeddings to be the worst, because it cannot learn the embeddings and it was trained on English data.

![FREEZE](img/Freeze_emb_eval_loss.svg?raw=True "Losses for different models")

![FREEZE](img/Freeze_emb_f1.svg?raw=True "F1 scores for different models")

From the graphs above, we can see, that my expectations were correct.

What is more, we can see that the BERT runs had a lot of variance, which is probably caused by the fact, that the model was not trained on Czech data, so it was kind of "guessing".
CZERT runs on the other hand were practically the same, very stable and very good.

The freezing of the embeddings did not have a big impact, I expected the gap between the two curves to be bigger.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

2. Do you see any result improvement/impairment when freezing the lower layers of the CZERT model?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I would expect the model to perform worse for frozen lower layers, but I do not really have a good reason for that --
just an intuition; I also do not have much experience with this, so I cannot really say for sure.

![FREEZE](img/Freeze_first_eval_loss.svg?raw=True "Losses for different models")

![FREEZE](img/Freeze_first_f1.svg?raw=True "F1 scores for different models")

From the loss graph, we can see that the model with 0 frozen lower layers performed the worst!
On the F1 graph, all the models seem to be more or less the same (the losses were not that different either).
So the answer is no, I do not see any result improvement / impairment when freezing the lower layers of the CZERT model.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

3. Does freezing the lower layers bring any benefits in term of results, speed of training, etc?

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

I would expect the frozen models to train faster, because they have less parameters to train.

![FREEZE](img/Freeze_first_runtime.svg?raw=True "Losses for different models")

The graph above does not support my intuition at all. In fact the most frozen model has the highest runtime.
This is really odd and I do not know how to interpret / explain this.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

4. Does the BERT model work for Czech tasks? State the results and include a graph of eval metrics for the BERT model config for both tasks.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer begin`

BERT does not work well for Czech tasks, at least not with the hyperparameters I tried to tune on the tasks I have tried to solve.

![BERT](img/BERT_more_like_FART_f1_parallel.svg?raw=True "F1 scores for different models")
(raw runs)

![BERT](img/BERT_more_like_FART_f1_parallel_grouped_by.svg?raw=True "F1 scores for different models")
(grouped by model type)

On the graphs above, we can see that the BERT model was not able to achieve good results on the Czech tasks.

![#000800](https://placehold.co/15x15/008000/008000.png) `Answer end`

## Questions to think about (test preparation, better understanding):

1. Why CZERT and Slavic works when embeddings are freezed and RNN and LSTM model strugles in this setup?
2. Describe the benefits of subword tokenization?
3. Does W2V uses whole word od subword tokenization?
4. Name 3 real world use cases for NER?
5. How is the morphological tagging different from NER? Can we use the same model? If not, what would you change?
6. What is the benefit of using BiLSTM instead of unidirectional LSTM?
7. Is the dataset balanced or unbalanced? Why can it happen that a model can learn to always output the majority class in case of unbalanced classification?
8. Why can the bi-directionality of the LSTM help to solve taks such as NER or TAGGING?
9. How did you compute the L2 norm. Which weights did/didn't you used and why?
10. How are the following metrics calculated: F1, precission, recall. What is the difference between macro and micro averaging with when computing the F1?
11. Explain why F1=precision=recall when using micro averaging?
12. Can we use to predictions from tagging model to improve the named entity recognition model? If so, please describe how would you do that?

