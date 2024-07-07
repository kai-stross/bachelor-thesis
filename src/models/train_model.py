import torch
import time
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def train(data, model, optimizer, criterion):
    """
    Train the model for one epoch on the training set.

    Args:
    data (torch_geometric.data.Data): The dataset containing features, edge indices, and train mask.
    model (torch.nn.Module): The model to be trained.
    optimizer (torch.optim.Optimizer): The optimizer for training the model.
    criterion (torch.nn.Module): The loss function.

    Returns:
    float: The training loss for the epoch.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    
    # Check if train_mask is set correctly
    if data.train_mask is None or data.train_mask.sum() == 0:
        print("Training mask is not set correctly.")
        return float('inf')

    optimizer.zero_grad()
    output = model(data.x.to(device), data.edge_index.to(device))
    loss = criterion(output[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    
    return loss.item()

def validate(data, model):
    """
    Validate the model on the validation set.

    Args:
    data (torch_geometric.data.Data): The dataset containing features, edge indices, and validation mask.
    model (torch.nn.Module): The trained model to be validated.

    Returns:
    tuple: A tuple containing:
        - accuracy (float): The accuracy of the model on the validation set.
        - precision (float): The precision of the model on the validation set.
        - recall (float): The recall of the model on the validation set.
        - f1 (float): The F1 score of the model on the validation set.
        - probs (torch.Tensor): The probabilities of the positive class for each validation node.
        - labels (torch.Tensor): The true labels of the validation nodes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        val_mask = data.val_mask
        
        if val_mask is None or val_mask.sum() == 0:
            print("Validation mask is not set correctly.")
            return 0, 0, 0, 0, None, None
        
        probs = torch.softmax(out[val_mask], dim=1)[:, 1]
        preds = out[val_mask].argmax(dim=1)
        labels = data.y[val_mask].to(device)

        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

    return accuracy, precision, recall, f1, probs, labels

def train_validate(models, csv_file, num_runs=1, early_stopping_patience=100, min_epochs=200, plot_training=True, print_epochs=True, plot_metric='f1'):
    """
    Train and validate multiple models with the given configurations, and save the results to a CSV file.

    Args:
    models (list): A list of model configurations.
    csv_file (str): The path to the CSV file to save results.
    num_runs (int): The number of runs for each model.
    early_stopping_patience (int): The patience for early stopping.
    min_epochs (int): The minimum number of epochs before early stopping can occur.
    plot_training (bool): Whether to plot the training process.
    print_epochs (bool): Whether to print epoch details.
    plot_metric (str): The metric to plot during training; valid options are 'loss', 'accuracy', 'precision', 'recall', 'f1'.

    Returns:
    list: A list of trained models and their validation metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_models = []
    model_count = 1
    model_amount = len(models)

    csv_columns = [
        'Time Stamp', 'Model Name', 'Dataset', 'Number of Runs', 'Computation', 'Std Computation', 'Epoch',
        'Number of Features', 'Hidden Channels', 'Learning Rate', 'Weight Decay',
        'Dropout Rate', 'Criterion', 'Loss', 'Accuracy', 'Precision',
        'Recall', 'F1', 'Std Loss', 'Std Accuracy', 'Std Precision', 'Std Recall', 'Std F1'
    ]

    write_header = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0

    if write_header:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_columns)

    for config in models:
        model_name = config['model_name']
        model_class = config['model']
        data_name = config['data_name']
        data = config['data']
        hidden_channels = config['hidden_channels']
        learning_rate = config['learning_rate']
        num_epochs = config['num_epochs']
        weight_decay = config['weight_decay']
        dropout_rate = config['dropout_rate']
        criterion = config['criterion']
        
        avg_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        metrics_list = {'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        total_computation_time = 0
        computation_times = []
        
        print(f"\nTraining {model_name} model ({model_count}/{model_amount}) on {data_name} dataset...")
        for run in range(num_runs):           
            model = model_class(num_features=data.num_features, out_channels=hidden_channels, dropout_rate=dropout_rate).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            start_time = time.time()
            time_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
            best_val_f1 = 0
            early_stopping_counter = 0
            real_epochs = 0

            epoch_iter = range(num_epochs)
            if not print_epochs:
                epoch_iter = tqdm(epoch_iter, desc=f'Run {run+1}/{num_runs}, {model_name}')

            run_train_losses = []
            run_val_losses = []
            run_val_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

            for epoch in epoch_iter:
                real_epochs += 1
                loss = train(data, model, optimizer, criterion)
                val_accuracy, val_precision, val_recall, val_f1, _, _ = validate(data, model)
                val_loss = criterion(model(data.x.to(device), data.edge_index.to(device))[data.val_mask], data.y[data.val_mask].to(device)).item()

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping_patience and epoch >= min_epochs:
                    if print_epochs:
                        print(f'Early stopping at epoch {epoch+1}')
                    else:
                        epoch_iter.set_description(f'Run {run+1}/{num_runs}, {model_name} (Early stop)')
                    break

                run_train_losses.append(loss)
                run_val_losses.append(val_loss)
                run_val_metrics['accuracy'].append(val_accuracy)
                run_val_metrics['precision'].append(val_precision)
                run_val_metrics['recall'].append(val_recall)
                run_val_metrics['f1'].append(val_f1)

                if print_epochs and (epoch + 1) % 50 == 0:
                    print(f'Run: {run + 1}/{num_runs}, Model: {model_name}, Epoch: {epoch+1}/{num_epochs}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
            
            end_time = time.time()
            computation_time = end_time - start_time
            total_computation_time += computation_time
            computation_times.append(computation_time)

            avg_metrics['loss'] += loss
            avg_metrics['accuracy'] += val_accuracy
            avg_metrics['precision'] += val_precision
            avg_metrics['recall'] += val_recall
            avg_metrics['f1'] += val_f1

            metrics_list['loss'].append(loss)
            metrics_list['accuracy'].append(val_accuracy)
            metrics_list['precision'].append(val_precision)
            metrics_list['recall'].append(val_recall)
            metrics_list['f1'].append(val_f1)

            if plot_training:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.plot(run_train_losses, label='Training Loss')
                plt.plot(run_val_losses, label='Validation Loss')
                plt.title(f'{model_name} Training (Run {run+1})')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(run_val_metrics[plot_metric], label=f'Validation {plot_metric.capitalize()}')
                plt.xlabel('Epochs')
                plt.ylabel(plot_metric.capitalize())
                plt.legend()
                plt.show()

        for key in avg_metrics:
            avg_metrics[key] /= num_runs

        std_metrics = {key: np.std(metrics_list[key]) for key in metrics_list}

        avg_computation_time = total_computation_time / num_runs
        std_computation_time = np.std(computation_times)
        avg_computation_time_str = f'{avg_computation_time // 60:.0f}m {avg_computation_time % 60:.0f}s'

        print(f'Successfully trained {model_name} model ({model_count}/{model_amount}) on {data_name} dataset')
        model_count += 1
        
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time_stamp, model_name, data_name, num_runs, avg_computation_time_str, std_computation_time, real_epochs, data.num_features, hidden_channels, learning_rate, weight_decay, dropout_rate, type(criterion).__name__, avg_metrics['loss'], avg_metrics['accuracy'], avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1'], std_metrics['loss'], std_metrics['accuracy'], std_metrics['precision'], std_metrics['recall'], std_metrics['f1']])

        trained_models.append({'model_name': model_name, 'model': model, 'data_name': data_name, 'data': data, 'learning_rate': learning_rate, 'dropout_rate': dropout_rate, 'weight_decay': weight_decay, 'accuracy': avg_metrics['accuracy'], 'precision': avg_metrics['precision'], 'recall': avg_metrics['recall'], 'f1': avg_metrics['f1'], 'computation': avg_computation_time_str})
    
    return trained_models
