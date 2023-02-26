# Claimer Detection using BERT-based Language Models

This repository contains code for a machine learning model that predicts the claimer of a claim. The model has been trained on four BERT-based language models: RoBERTa, BART, ALBERT, and DistilBERT.

## How to Train the Model
To train the model, follow the steps below:

- Clone this repository to your local machine.

- Navigate to the cloned directory.

- Install the required dependencies using pip install -r requirements.txt.

- Download the annotated data to be used for training and evaluation and store them in the annotated/ directory.

- Run the following command to start training the model:
```
sudo python3 train.py \
    --model_type <model name, ex roberta,bert etx> \
    --model_name_or_path <path to the model, could be an existing model or local fodler with a model> \
    --output_dir <output dir for the mdoel> \
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
  
## How to Evaluate the Model

After you have succesfullt do the above steps (cloning, instal etc) :

- Run the following command to start evaluating the model:

```
sudo python3 evaluate.py \
    --model_name_or_path <path to the model, could be an existing model or local fodler with a model> \
    --output_dir <output dir for the mdoel> \
    --data_dir <folder with the data> \
    --do_lower_case \
    --predict_file <file for evaluation/prediction> \
```

## How to make Predictions
Once the model is trained or if you already have a model, you can use it to make predictions by following the steps below:
- Make sure you have the necessary libraries installed by running pip install -r requirements.txt.

- Run the following command to make a prediction:
```
python3 predict.py \
    --context <the context you want to make a prediction on> \
    --claim <the claim what you want to find the claimer> \
    --model_name_or_path <model name or path to the model>
```

The returned value will be the Claimer of the claim. If the value that the model returns is not an organisation or a person, or the predicted propability is bellow 0.01 then the reurned value will be 'Author' meaning the Author of the context.
