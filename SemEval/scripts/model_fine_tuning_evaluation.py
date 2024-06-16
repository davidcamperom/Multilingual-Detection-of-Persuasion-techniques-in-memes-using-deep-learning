# This script takes a different PREPROCESSED training dataset (original train and balanced train) and use them as input to fine-tune different LLM models.
# Make sure to execute this python script in the SemEval directory

import json
import argparse
import random
import torch
import numpy as np
import os
import subprocess
import shutil
import tensorflow as tf
import fwr13y.d9m.tensorflow as tf_determinism
from pytorch_lightning import seed_everything
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer # to one-hot-encode the label columns
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline, EarlyStoppingCallback
from transformers import set_seed, enable_full_determinism
from transformers import GPT2Tokenizer, GPT2LMHeadModel # to fine tune GPT2
from transformers import DataCollatorWithPadding

# Para garantizar la reproducibilidad de nuestros experimentos
# Establecer el determinismo
tf_determinism.enable_determinism()
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)# Store the average loss after eachepoch so we can plot them.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["TF_DETERMINISTIC_OPS"] = "1" # See:https://github.com/NVIDIA/tensorflow-determinism#confirmed-current-gpu-specific-sources-of-non-determinism-with-solutions
set_seed(seed_val)
enable_full_determinism(seed_val)
seed_everything(seed_val, workers=True)

# set default plot parameters
plt.rcParams['axes.spines.top'] = False # remove top spine
plt.rcParams['axes.spines.right'] = False # remove right spine
plt.rcParams['pdf.fonttype'] = 42 # Set the fonttype to export fonts as font files
plt.rcParams['font.family'] = 'Arial'

# set up root path
root_path = os.getcwd()

# path to the datasets
data_path = f'{root_path}/data/'

# path to the models
model_path = f'{root_path}/fine_tuned_models/'

# path to save figures
save_path = f'{root_path}/figures/'

# keep a label variable as it is handy
labels = ['Appeal to authority', 'Appeal to fear/prejudice', 'Bandwagon',
          'Black-and-white Fallacy/Dictatorship', 'Causal Oversimplification',
          'Doubt', 'Exaggeration/Minimisation', 'Flag-waving',
          'Glittering generalities (Virtue)', 'Loaded Language',
          "Misrepresentation of Someone's Position (Straw Man)",
          'Name calling/Labeling',
          'Obfuscation, Intentional vagueness, Confusion',
          'Presenting Irrelevant Data (Red Herring)', 'Reductio ad hitlerum',
          'Repetition', 'Slogans', 'Smears', 'Thought-terminating cliché',
          'Whataboutism']
# order the labels by abundance in the original (imbalance) training data
label_order = ['Smears','Loaded Language','Name calling/Labeling','Appeal to authority',
               'Black-and-white Fallacy/Dictatorship','Slogans','Flag-waving',
               'Thought-terminating cliché','Glittering generalities (Virtue)',
               'Exaggeration/Minimisation','Doubt','Appeal to fear/prejudice','Repetition',
               'Whataboutism','Causal Oversimplification','Bandwagon','Reductio ad hitlerum',
               "Misrepresentation of Someone's Position (Straw Man)",
               'Presenting Irrelevant Data (Red Herring)',
               'Obfuscation, Intentional vagueness, Confusion']
# model parameters
BATCH_SIZE = 8 # was 32, changed to this after hyperparam tuning with optuna
NUM_TRAIN_EPOCHS = 5 # was 10, changed to this after hyperparam tuning with optuna
# NUM_TRAIN_EPOCHS = 1 # set this for faster fine tuning during testing
LEARNING_RATE = 4.4e-05 # was 5e-5, changed to this after hyperparam tuning with optuna
MAX_LENGTH = 128
WEIGHT_DECAY = 0.01

def prepare_dataset(dataframe):
    """
    Converts pandas dataframes into Dataset objects that are compatible with torch and further 
    convert the labels into arrays of 1s and 0s for model fine-tuning

    Args:
        dataframe (pd.DataFrame): pandas dataframe to be converted to Dataset object
    
    Returns:
        dataset (Dataset): The converted Dataset object
    """
    dataset = Dataset.from_pandas(dataframe)
    # convert the labels into arrays of 1s and 0s
    dataset = dataset.map(lambda x: {"labels": [x[label] for label in x if label in labels]})

    # convertimos los datasets a formato de torch y tambien sus etiquetas
    dataset.set_format("torch")
    dataset = (dataset.map(lambda x : {"float_labels": x["labels"].to(torch.float)},
                           remove_columns=["labels"]).rename_column("float_labels", "labels"))
    
    return dataset

def tokenize_dataset(dataset,model_checkpoint):
    """
    Tokenize datasets by using AutoTokenizer of model_checkpoint

    Args:
        dataset (Dataset): Dataset object formatted to be compatible with torch
        model_checkpoint (str): Model checkpoint, e.g. 'google-bert/bert-base-uncased'

    Returns:
        tokenizer (AutoTokenizer): The used tokenizer
        model (AutoModelForSequenceClassification): The instantiated model
        dataset_encoded (Dataset): Dataset with only columns needed for model fine-tuning
    """
    def tokenize_data(dataset):
        """ Tokenizes the data in a dataset and returns the output of the tokenizer

        Args:
            text (str): Input text

        Returns:
            outputs of tokenizer
        """
        # if 'gpt' in model_checkpoint:
        #     return tokenizer(dataset['processed_text'], padding='max_length', truncation=True, max_length=512)
        return tokenizer(dataset['processed_text'], truncation=True, max_length=MAX_LENGTH, padding=True)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=len(labels),problem_type="multi_label_classification")
    data_collator = DataCollatorWithPadding(tokenizer)

    if model_checkpoint == 'openai-community/gpt2': # for fine-tuning purpose, gpt2 model requires pad_token to be manually set
        print('GPT2 specific modifications!')
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    remove_columns = dataset.column_names  # Take all columns
    remove_columns.remove("labels") # Remove the column "labels" so that it does not get removed

    data_encoded = dataset.map(tokenize_data,batched=True,remove_columns=remove_columns)

    return tokenizer, model, data_collator, data_encoded

def fine_tune_model(train_dataset_encoded,validation_dataset_encoded,model_checkpoint,train_df_name,model,tokenizer,data_collator):
    """
    Takes a train and validation dataset (formatted to be compatible with torch and encoded (tokenized))
    to fine-tune a provided LLM

    Args:
        train_dataset_encoded (Dataset): Training Dataset with only columns needed for model fine-tuning
        validation_dataset_encoded (Dataset): Validation Dataset with only columns needed for model fine-tuning
        model_checkpoint (str): Model checkpoint, e.g. 'google-bert/bert-base-uncased'
        train_df_name (str): An arbitrary name given to train_df to save the fine-tuned model in folder named after the train_df_name
        model (AutoModelForSequenceClassification): The instantiated model from tokenize_dataset()
        tokenizer (AutoTokenizer): The instantiated tokenizer from tokenize_dataset()
        data_collator (DataCollatorWithPadding): The instantiated DataCollatorWithPadding from tokenize_dataset() to enable dynamic padding

    Returns:
        trainer (Trainer): The trainer object containing the fine-tuned LLM
        out_path (str): Path to where the fine-tuned model is saved
        label_optim_threshold (dict): Optimal thresholds of individual labels derived using the dev_dataset
    """
    # set up a directory to store the fine-tuned model and its output files
    out_path = f'{model_path}{model_checkpoint}/{train_df_name}/'
    out_path_fig = f'{out_path}figures/'
    out_path_eval = f'{out_path}eval/'
    if os.path.exists(out_path): # if exists, remove and create a new one (overwrite)
        shutil.rmtree(out_path)
        os.makedirs(out_path)
    else:
        os.makedirs(out_path)
    # make a figures dir in the out_path to keep the plots of model evaluation
    os.mkdir(out_path_fig)
    os.mkdir(out_path_eval)

    # def model_init():
    #     return AutoModelForSequenceClassification.from_pretrained(model_checkpoint,num_labels=len(labels),problem_type="multi_label_classification")
    
    def multi_label_metrics(labels,predictions,threshold=None):
        """
        Function to evaluate a fine-tuned model using different metrics
        """
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        y_true = labels
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        probs = probs.numpy()

        # the calculation of ROC curve and PR curve AUC do not require thresholded probabilities
        roc_auc = roc_auc_score(y_true, probs, average = 'weighted') # LCY: should be weighted because each label has different number of support
        ap = average_precision_score(y_true, probs, average='weighted')

        # if threshold is None, use an array of increasing threshold to apply on the probs to exhaustively find the optimal threshold that yields the max metrics
        if threshold is None:
            thresholds = np.arange(0, 1, step=0.02)
            f1 = []
            acc = []
            # binarize the predictions probs to 1 and 0 by applying the threshold iteratively on them
            for threshold in thresholds:
                y_pred = np.where(probs >= threshold, 1, 0)
                f1.append(f1_score(y_true=y_true,y_pred=y_pred,average='weighted'))
                acc.append(accuracy_score(y_true=y_true,y_pred=y_pred))
        # if threshold provided, use that threshold to compute the metrics
        else:
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= threshold)] = 1
            f1 = f1_score(y_true=y_true,y_pred=y_pred,average='weighted')
            acc = accuracy_score(y_true=y_true,y_pred=y_pred)

        # return as dictionary
        metrics = {
            'roc_auc': roc_auc,
            'ap': ap,
            'max_f1_score': max(f1) if isinstance(f1,list) else f1,
            'max_accuracy': max(acc) if isinstance(acc,list) else acc,
            }
        return metrics

    def compute_metrics(eval_pred):
        """
        Function to be provided to Trainer object for evaluation of fine-tuned models
        """
        preds = eval_pred.predictions[0] if isinstance(eval_pred.predictions,
                tuple) else eval_pred.predictions
        result = multi_label_metrics(
            labels=eval_pred.label_ids,
            predictions=preds,
            )
        return result
    
    # set up TrainingArguments
    logging_steps = len(train_dataset_encoded) // (2 * BATCH_SIZE * NUM_TRAIN_EPOCHS)
    print("********************** logging_steps", logging_steps)
    optim=["adamw_hf", "adamw_torch", "adamw_apex_fused","adafactor","adamw_torch_xla"]

    training_args = TrainingArguments(
        output_dir = 'results',
        num_train_epochs = NUM_TRAIN_EPOCHS,
        learning_rate = LEARNING_RATE,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size = BATCH_SIZE,
        load_best_model_at_end = True,
        #metric_for_best_model = 'f1',
        metric_for_best_model = 'ap', # Use average precision because it is more suited for classification tasks with class imbalance issue
        #metric_for_best_model = 'eval_loss',
        greater_is_better=True, # tells the model to take greater metric_for_best_model as better model
        weight_decay = WEIGHT_DECAY,
        evaluation_strategy = 'epoch',
        save_strategy = 'epoch',
        logging_steps = logging_steps,
        save_total_limit = 3,
        optim = optim[1],
        push_to_hub=False
        )
    
    # set up Trainer and execute model fine-tuning
    trainer = Trainer(
        model=model,
        # model_init=model_init,
        args = training_args,
        compute_metrics = compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)],
        train_dataset = train_dataset_encoded,
        eval_dataset = validation_dataset_encoded,
        data_collator = data_collator,
        tokenizer = tokenizer
        )
    
    torch.cuda.empty_cache()
    trainer.train()
    
    # save the fine-tuned LLM
    trainer.save_model(out_path)

    # plot loss over epoch and save
    trainer_history = pd.DataFrame(trainer.state.log_history)
    fig, ax = plt.subplots(sharex=True)
    sns.lineplot(data=trainer_history,x='epoch',y='loss',linestyle='-',ax=ax)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    # twin the x axis to have another y axis on the right to plot the learning_rate
    ax2 = ax.twinx()
    sns.lineplot(data=trainer_history,x='epoch',y='learning_rate',linestyle='--',ax=ax2)
    ax2.spines[['right']].set_visible(True)
    ax2.set_ylabel('Learning rate')

    h = ax.get_lines()
    h.extend(ax2.get_lines())
    plt.legend(handles=h, labels=['Loss','Learning rate'], loc='best')
    plt.savefig(f"{out_path_fig}loss_learning_rate_over_epoch.pdf",transparent=True,bbox_inches='tight')
    plt.savefig(f"{out_path_fig}loss_learning_rate_over_epoch.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

    eval = trainer.evaluate()
    dfeval = pd.DataFrame(list(eval.items()), columns = ['Name','Validation_value'])
    dfeval.to_csv(f'{out_path_eval}validation_set_eval.csv')

    # evaluate the fine-tuned model using the validation set
    # Utilizar la funcion Sigmoid para convertir los logits (raw output) del modelo a probabilidades (en el rango de 0 a 1)
    sigmoid = torch.nn.Sigmoid()

    # Logits are stored in predictions.predictions
    predictions = trainer.predict(validation_dataset_encoded)
    logits = predictions.predictions
    probs = sigmoid(torch.Tensor(logits))
    probs = probs.numpy()
    
    # make the ROC_AUC and AP barplots
    roc_auc = roc_auc_score(y_true=predictions.label_ids, y_score=probs, average = None)
    ap = average_precision_score(y_true=predictions.label_ids, y_score=probs, average = None)
    ROC_PR_per_label = pd.DataFrame({'Label':labels,'ROC_AUC':roc_auc,'AP':ap})
    ROC_PR_per_label.to_csv(f'{out_path_eval}validation_set_ROC_AUC_AP.csv')

    # plot the barplots
    temp = ROC_PR_per_label.melt(id_vars='Label',value_vars=['ROC_AUC','AP'])
    fig, ax = plt.subplots()
    sns.barplot(data=temp,x='value',y='Label',hue='variable',orient='h',order=label_order,ax=ax)
    ax.legend()
    plt.savefig(f"{out_path_fig}validation_ROC_AUC_AP.pdf",transparent=True,bbox_inches='tight')
    plt.savefig(f"{out_path_fig}validation_ROC_AUC_AP.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

    # find the optimal thresholds for each label by finding the thresholds that give the highest
    # f1 score and apply them to classify the probs
    thresholds = np.arange(0,1,step=0.02)

    label_optim_threshold = {}

    for i in range(len(labels)):
        f1 = []
        label_probs = probs[:,i]
        label_truth = predictions.label_ids[:,i]
        for threshold in thresholds:
            y_pred = np.where(label_probs >= threshold, 1, 0)
            f1.append(f1_score(y_true=label_truth,y_pred=y_pred,average='weighted'))
        max_f1_id = np.argmax(f1)
        label_optim_threshold[labels[i]] = thresholds[max_f1_id]

    # save the optimal thresholds
    with open(f'{out_path}validation_set_label_optimal_thresholds.json','w') as f:
        json.dump(label_optim_threshold,f,indent=4)

    return trainer, out_path, label_optim_threshold

def evaluate_model_dev(trainer,dev_df_ind,dev_dataset_encoded,out_path,label_optim_threshold):
    """
    Evaluate a fine-tuned model using a provided development dataset through ROC and PR curve analyses 
    and derive the optimal thresholds to classify the predicted probabilities as 1 or 0 using 
    the development dataset

    Args:
        trainer (Trainer): The fine-tuned model saved as Trainer object
        dev_df_ind (list of str): List of indices from dev_df for the ouput of predicted labels
        dev_dataset_encoded (Dataset): Development Dataset with only columns needed for model evaluation
        out_path (str): Path to where the fine-tuned model is saved
        label_optim_threshold (dict): Optimal thresholds of individual labels derived using the dev_dataset
    
    Returns:
        None
    """
    out_path_fig = f'{out_path}figures/'
    out_path_eval = f'{out_path}eval/'

    dev = trainer.evaluate(dev_dataset_encoded)
    dfdev = pd.DataFrame(list(dev.items()), columns = ['Name','Development_value'])
    dfdev.to_csv(f'{out_path_eval}dev_set_eval.csv')

    sigmoid = torch.nn.Sigmoid()

    # Logits are stored in predictions.predictions
    predictions = trainer.predict(dev_dataset_encoded)
    logits = predictions.predictions
    probs = sigmoid(torch.Tensor(logits))
    probs = probs.numpy()
    
    roc_auc = roc_auc_score(y_true=predictions.label_ids, y_score=probs, average = None)
    ap = average_precision_score(y_true=predictions.label_ids, y_score=probs, average = None)

    ROC_PR_per_label = pd.DataFrame({'Label':labels,'ROC_AUC':roc_auc,'AP':ap})
    ROC_PR_per_label.to_csv(f'{out_path_eval}dev_set_ROC_AUC_AP.csv')

    # plot the ROC_AUC and AP as barplot
    temp = ROC_PR_per_label.melt(id_vars='Label',value_vars=['ROC_AUC','AP'])
    fig, ax = plt.subplots()
    sns.barplot(data=temp,x='value',y='Label',hue='variable',orient='h',order=label_order,ax=ax)
    ax.legend()
    plt.savefig(f"{out_path_fig}dev_ROC_AUC_AP.pdf",transparent=True,bbox_inches='tight')
    plt.savefig(f"{out_path_fig}dev_ROC_AUC_AP.png",dpi=300,bbox_inches='tight')
    plt.close(fig)

    # classify the probabilities into labels and save into an output file
    pred_labels = convert_probs_to_labels(pred_probs=probs,label_optim_threshold=label_optim_threshold) # IMPORTANT: The optimal thresholds are derived from validation set and applied on this dev set
    
    dev_pred_dict = {}
    for ind, pred_label in zip(dev_df_ind,pred_labels):
        label_list = []
        for label, pred in zip(labels,pred_label):
            if pred == 1:
                label_list.append(label)
        dev_pred_dict[ind] = label_list

    with open(f'{out_path}dev_predicted_labels.json', 'w') as f:
        json.dump([{'id': str(k), 'labels': v} for k, v in dev_pred_dict.items()], f, indent=4)

def convert_probs_to_labels(pred_probs,label_optim_threshold):
    """
    Use a dictionary of labels with their optimal thresholds to call predicted probabilities as 
    1s or 0s. The order of the labels in label_optim_threshold is important as it was saved 
    in the same order as the predicted probabilities!

    Args:
        pred_probs (np.array): NumPy array of nrow x 20 predicted probabilities (for each label)
        label_optim_threshold (dict): Optimal thresholds of individual labels derived using the dev_dataset

    Returns:
        pred_labels (np.array): NumPy array of nrow x 20 predicted label (for each label)
    """
    pred_labels = np.zeros(pred_probs.shape)

    for i, ele in enumerate(label_optim_threshold):
        optim_threshold = label_optim_threshold[ele]
        pred_labels[:,i] = np.where(pred_probs[:,i] >= optim_threshold, 1, 0)
    
    return pred_labels

def evaluate_model_test(tokenizer,trainer,test_df,label_optim_threshold,out_path):
    """
    Takes a test dataframe with 'processed_text' column and convert it into Dataset object for 
    tokenization. Use a fine-tuned model to predict the encoded test dataset and apply the optimal 
    thresholds derived from development dataset to classify the predicted probabilities as 1 or 0. 
    The ids and their predicted labels are saved in the provided out_path

    Args:
        tokenizer (AutoTokenizer): The used tokenizer
        trainer (Trainer): The fine-tuned model saved as Trainer object
        test_df (pd.DataFrame): The test dataset as a pandas dataframe
        label_optim_threshold (dict): Optimal thresholds of individual labels derived using the dev_dataset
        out_path (str): Path to where the fine-tuned model is saved in order to save the ids and
        their predicted labels
    
    Returns:
        None
    """
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_data(dataset):
        """ Tokenizes the data in a dataset and returns the output of the tokenizer

        Args:
            text (str): Input text

        Returns:
            outputs of tokenizer
        """
        return tokenizer(dataset['processed_text'], truncation=True, max_length=MAX_LENGTH, padding=True)

    remove_columns = ['text','labels','link',]
    test_dataset_encoded = test_dataset.map(tokenize_data, batched=True, remove_columns=remove_columns)
    
    sigmoid = torch.nn.Sigmoid()

    # Logits are stored in predictions.predictions
    predictions = trainer.predict(test_dataset_encoded)
    logits = predictions.predictions
    probs = sigmoid(torch.Tensor(logits))
    probs = probs.numpy()

    pred_labels = convert_probs_to_labels(pred_probs=probs,label_optim_threshold=label_optim_threshold) # IMPORTANT: The optimal thresholds are derived from validation set and applied on this test set
    
    test_pred_dict = {}
    for ind, pred_label in zip(test_df.index,pred_labels):
        label_list = []
        for label, pred in zip(labels,pred_label):
            if pred == 1:
                label_list.append(label)
        test_pred_dict[ind] = label_list

    with open(f'{out_path}test_predicted_labels.json', 'w') as f:
        json.dump([{'id': str(k), 'labels': v} for k, v in test_pred_dict.items()], f, indent=4)

def main():
    """
    Wrapper function to execute the script.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_df_name",'-train_name',type=str,required=True,help="An arbitrary name to give to the training dataset")
    parser.add_argument("--train_df_file",'-train',type=str,required=True,help="Paths to the preprocessed training dataset")
    parser.add_argument("--validation_df_file",'-val',type=str,required=True,help="Paths to the preprocessed validation dataset")
    parser.add_argument("--dev_df_file",'-dev',type=str,required=True,help="Paths to the preprocessed development dataset")
    parser.add_argument("--test_df_file",'-test',type=str,required=True,help="Paths to the preprocessed test dataset")
    args = parser.parse_args()
    
    train_df_name = args.train_df_name 
    train_df = pd.read_csv(args.train_df_file,index_col=0).fillna({'processed_text':''}) 
    validation_df = pd.read_csv(args.validation_df_file,index_col=0).fillna({'processed_text':''})
    dev_df = pd.read_csv(args.dev_df_file,index_col=0).fillna({'processed_text':''})
    test_df = pd.read_csv(args.test_df_file,index_col=0).fillna({'processed_text':''})
    # fillna because some processed_text was preprocessed into empty string and they get saved as NaN
    
    dev_df_ind = dev_df.index # to save the predicted labels as a json file

    model_checkpoints = [
        'google-bert/bert-base-uncased',
        'microsoft/deberta-v3-base',
        'FacebookAI/roberta-base',
        'distilbert/distilbert-base-uncased',
        'xlnet/xlnet-base-cased',
        'openai-community/gpt2',
        ]
    
    for model_checkpoint in model_checkpoints:
        if os.path.exists(f'{model_path}{model_checkpoint}/{train_df_name}/config.json'):
            print(f'{model_checkpoint} fine-tuning using {train_df_name} already performed, skipping this model!')
            continue
        print(f'Fine-tuning {model_checkpoint}!')
        train_dataset = prepare_dataset(train_df)
        tokenizer, model, data_collator, train_dataset_encoded = tokenize_dataset(train_dataset,model_checkpoint)
        
        validation_dataset = prepare_dataset(validation_df)
        tokenizer, model, data_collator, validation_dataset_encoded = tokenize_dataset(validation_dataset,model_checkpoint)

        dev_dataset = prepare_dataset(dev_df)
        tokenizer, model, data_collator, dev_dataset_encoded = tokenize_dataset(dev_dataset,model_checkpoint)

        trainer, out_path, label_optim_threshold = fine_tune_model(train_dataset_encoded,validation_dataset_encoded,model_checkpoint,train_df_name,model,tokenizer,data_collator)
        evaluate_model_dev(trainer,dev_df_ind,dev_dataset_encoded,out_path,label_optim_threshold)
        evaluate_model_test(tokenizer,trainer,test_df,label_optim_threshold,out_path)

        # perform the SemEval hierarchical evaluation on the predicted labels and save the output in eval folder
        file_ = open(f'./fine_tuned_models/{model_checkpoint}/{train_df_name}/eval/dev_hierarchical_f1.txt', "w")
        subprocess.Popen(f'python3 ./data/scorer-baseline/subtask_1_2a.py -g ./data/dev_gold_labels/dev_subtask1_en.json -p ./fine_tuned_models/{model_checkpoint}/{train_df_name}/dev_predicted_labels.json',shell=True,stdout=file_)
        file_.close()

        # os.system(f"""python3 ./data/scorer-baseline/subtask_1_2a.py -g ./data/dev_gold_labels/dev_subtask1_en.json -p {out_path}dev_predicted_labels.json > {out_path_eval}dev_hierarchical_f1.txt""")
        print(f'Fine-tuning {model_checkpoint} completed!\nThe fine-tuned model and its files are saved under {out_path}!')
    
if __name__ == '__main__':
    main()

# python3 ./scripts/model_fine_tuning_evaluation.py -train_name original_train -train ./data/preprocessed_datasets/train_df.csv -val ./data/preprocessed_datasets/validation_df.csv -dev ./data/preprocessed_datasets/dev_df.csv -test ./data/preprocessed_datasets/test_df.csv