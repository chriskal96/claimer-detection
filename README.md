# Claimer Detection using BERT-based Language Models

This repository contains code for a machine learning model that predicts the claimer of a claim. The model has been trained on four BERT-based language models: RoBERTa, BART, ALBERT, and DistilBERT.


## Data
To train and evaluate the models 3 datasets were used :
 - Stanford Question Answering Dataset (SQUAD)
 - Google's Natural Question Dataset (GNQ)
 - A new annotated dataset regarding food and health created as part of this project (not available to public)
 
### Download datasets :
 - For SQUAD, the train and eval datasets can be found here : https://rajpurkar.github.io/SQuAD-explorer/
 - For GNQ, instructions and the datasets can be found here : https://ai.google.com/research/NaturalQuestions/download 
 
For GNQ the simplified version was used. It was also transformed into the same format as SQUAD ([script](https://github.com/chriskal96/claimer-detection/blob/main/data/convert_gnq_to_squad.py)) in order to take advantage of already established libraries.
In folder [data](https://github.com/chriskal96/claimer-detection/tree/main/data) there are data exploration files for SQUAD 2.0, Google's Natural Question and a new annotated dataset used for testing the models.


## How to Train the Model
To train the model, follow the steps below:

- Clone this repository to your local machine.

- Navigate to the cloned directory.

- Install the required dependencies using pip install -r requirements.txt.

- Download the annotated data to be used for training and evaluation.

- Run the following command to start training the model:
```
python QaClaimer/train.py \
    --model_type <model name, e.g. roberta,bert etc> \
    --model_name_or_path <path to the model, could be an existing model or local folder with a model> \
    --output_dir <output directory for the new trained model> \
    --data_dir <folder with the data> \
    --train_file <file with trained data> \
    --do_lower_case \
    --predict_file <file for evaluation/prediction> \
    --batches_per_gpu_train <batch size per gpu> \
    --learning_rate <learning rate> \
    --num_train_epochs <number of training epochs> \
    --max_seq_length <mac sequence length> \
    --accept_answers_not_in_text <only if the model is going to be trained with answers not in the text>
   ```
 By setting accept_answers_not_in_text, the model can be trained to predict answers that are not in the context.
 
## How to Evaluate the Model

After you have succesfully execute the above steps (cloning, instal etc) :

- Run the following command to start evaluating the model:

```
python QaClaimer/evaluate.py \
    --model_name_or_path <path to the model, could be an existing model or local fodler with a model> \
    --output_dir <output dir for the mdoel> \
    --data_dir <folder with the data> \
    --do_lower_case \
    --predict_file <file for evaluation/prediction> \
```

## Results

The resulted F1 score by training 4 bert based models in the SQUAD dataset :

| Model   | Roberta | DistilBert | Albert | Bart   |
|---------|--------|------------|--------|--------|
| Score   | 83.46  | 68.58      | 80.87  | 79.54  |
 
The same models trained on the GNQ datasets :

| Model   | Roberta | DistilBert | Albert | Bart   |
|---------|--------|------------|--------|--------|
| Score   | 79.96  | 82.95      | 78.89  | 79.51  |

Training occured on a server of 2 GPUs with 12 batch size per GPU and a learning rate of 3e-05.
Experimenting with different hyperparameters did not increase the F1 scores.
The models that were created by training on those datasets were later fine tuned using the new annotated dataset (also transformed into SQUAD format).

## How to make Predictions
Once the model is trained or if you already have a model, you can use it to make predictions by following the steps below:
- Make sure you have the necessary libraries installed by running pip install -r requirements.txt.

- Run the following command to make a prediction:
```
python predict.py \
    --context <the context you want to make a prediction on> \
    --claim <the claim what you want to find the claimer> \
    --model_name_or_path <model name or path to the model>
```

The returned value will be the Claimer of the Claim. If the value that the model returns is not an organisation or a person (a named enitty recognition model was used to filter the model's output), or the predicted propability is bellow 0.01 (defined threshold for accepting predicted values) then the returned value will be 'Author' meaning the Author of the context.

## Acknowledgments

To train the models, already established libraries to preprocess the SQUAD dataset from HuggingFace Transformers were used, alterted in ways benefit to the project. A source of inspiration also was the run_squad script from HuggingFace Transformers.

## Author
<a href="https://github.com/chriskal96">Christos Kallaras</a><br/>
