import json
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold
from train_model import train, validate
import itertools
import pandas as pd
from train_model import train_validate
import itertools
import pandas as pd
import csv
from tqdm import tqdm
from skopt import gp_minimize
import torch.nn as nn
from torch.optim import Adam
import os
import time
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig
import networkx as nx
import matplotlib.pyplot as plt
from math import sqrt

def test(data, model):
    """
    Evaluate the model on the test set.

    Args:
    data (torch_geometric.data.Data): The dataset containing features, edge indices, and test mask.
    model (torch.nn.Module): The trained model to be evaluated.

    Returns:
    tuple: A tuple containing:
        - accuracy (float): The accuracy of the model on the test set.
        - precision (float): The precision of the model on the test set.
        - recall (float): The recall of the model on the test set.
        - f1 (float): The F1 score of the model on the test set.
        - probs (torch.Tensor): The probabilities of the positive class for each test node.
        - labels (torch.Tensor): The true labels of the test nodes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        test_mask = data.test_mask
        
        if test_mask is None or test_mask.sum() == 0:
            print("Test mask is not set correctly.")
            return 0, 0, 0, 0, None, None
        
        probs = torch.softmax(out[test_mask], dim=1)[:, 1]
        preds = out[test_mask].argmax(dim=1)
        labels = data.y[test_mask].to(device)

        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
        recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
        f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

    return accuracy, precision, recall, f1, probs, labels

def plot_roc(trained_models):
    """
    Plot ROC curves for multiple trained models.

    Args:
    trained_models (list): A list of model dictionaries, each containing 'model_name', 'model', and 'data' keys.

    Returns:
    None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.figure(figsize=(10, 8))
    
    for entry in trained_models:
        model_name = entry['model_name']
        model = entry['model']
        data = entry['data']
        
        model.eval()
        
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            probs = torch.softmax(out, dim=1)[:, 1].cpu()
            labels = data.y.to(device).cpu()
            
        fpr, tpr, thresholds = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_val_metrics(trained_models):
    """
    Plot validation metrics for multiple trained models.

    Args:
    trained_models (list): A list of dictionaries, each containing model metrics and details.

    Returns:
    None
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metrics_values = {metric: [] for metric in metrics}
    for model in trained_models:
        for metric in metrics:
            metrics_values[metric].append(model[metric])

    num_metrics = len(metrics) + 1
    num_models = len(trained_models)
    width = 0.15
    spacing = 0.4

    positions = []
    for i in range(num_metrics):
        base_position = i * (num_models * width + spacing)
        positions.extend([base_position + j * width for j in range(num_models)])

    fig, ax = plt.subplots(figsize=(12, 7))
    
    for idx, model in enumerate(trained_models):
        values = [model[metric] for metric in metrics]
        average_value = sum(values) / len(values)
        values.append(average_value)
        label = f"{model['model_name']} ({model['computation']})"
        bar_positions = [positions[i * num_models + idx] for i in range(num_metrics)]
        bars = ax.bar(bar_positions, values, width, label=label)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, round(yval, 4), ha='center', va='bottom')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Model Validation Metrics Comparison')

    metric_labels = metrics + ['metrics average']
    metric_positions = [(i * (num_models * width + spacing)) + (num_models * width) / 2 for i in range(num_metrics)]
    ax.set_xticks(metric_positions)
    ax.set_xticklabels(metric_labels)
    ax.legend()

    all_values = []
    for metric in metrics_values.values():
        all_values.extend(metric)
    for model in trained_models:
        avg_value = sum(model[metric] for metric in metrics) / len(metrics)
        all_values.append(avg_value)
        
    y_min = min(all_values) - 0.05
    y_max = max(all_values) + 0.05

    ax.set_ylim([y_min, y_max])

    fig.tight_layout()

    plt.show()

    for metric in metrics:
        std_dev = np.std(metrics_values[metric])
        print(f"Standard Deviation for {metric.capitalize()}: {std_dev:.3f}")

def grid_search(models, csv_file, learning_rates, dropout_rates, weight_decays, num_runs=1, early_stopping_patience=100, min_epochs=200, plot_training=False, print_epochs=False):
    """
    Perform a grid search over hyperparameters for training and validating models.

    Args:
    models (list): A list of model configurations.
    csv_file (str): The path to the CSV file to save results.
    learning_rates (list): A list of learning rates to try.
    dropout_rates (list): A list of dropout rates to try.
    weight_decays (list): A list of weight decay values to try.
    num_runs (int): The number of runs for each configuration.
    early_stopping_patience (int): The patience for early stopping.
    min_epochs (int): The minimum number of epochs before early stopping can occur.
    plot_training (bool): Whether to plot the training process.
    print_epochs (bool): Whether to print epoch details.

    Returns:
    list: A list of trained models and their validation metrics.
    """
    param_grid = list(itertools.product(learning_rates, dropout_rates, weight_decays))
    total_configs = len(param_grid)
    trained_models = []
    original_model_names = [model_config['model_name'] for model_config in models]
    
    for idx, (lr, dr, wd) in enumerate(param_grid, start=1):
        print(f"\nTraining configuration {idx}/{total_configs}: Learning Rate={lr}, Dropout Rate={dr}, Weight Decay={wd}...")
        
        for i, model_config in enumerate(models):
            model_config['model_name'] = f'{original_model_names[i]} LR={lr}, DR={dr}, WD={wd}'
            model_config['learning_rate'] = lr
            model_config['dropout_rate'] = dr
            model_config['weight_decay'] = wd
            
        trained_models.extend(train_validate(models, csv_file, num_runs, early_stopping_patience, min_epochs, plot_training, print_epochs))

        print(f"\nSuccessfully trained configuration {idx}/{total_configs}")
        
    return trained_models

def analyze_log_metrics(csv_file, num_rows=5):
    """
    Analyze and print top model metrics from a CSV log file.

    Args:
    csv_file (str): The path to the CSV file containing model metrics.
    num_rows (int): The number of top models to display for each metric.

    Returns:
    None
    """
    file = pd.read_csv(csv_file)
    file['Metrics Average'] = file[['Accuracy', 'Precision', 'Recall', 'F1']].mean(axis=1)

    best_accuracy = file['Accuracy'].max()
    best_precision = file['Precision'].max()
    best_recall = file['Recall'].max()
    best_f1 = file['F1'].max()
    best_metrics_avg = file['Metrics Average'].max()

    avg_accuracy = file['Accuracy'].mean()
    avg_precision = file['Precision'].mean()
    avg_recall = file['Recall'].mean()
    avg_f1 = file['F1'].mean()
    avg_metrics_avg = file['Metrics Average'].mean()

    std_accuracy = file['Accuracy'].std()
    std_precision = file['Precision'].std()
    std_recall = file['Recall'].std()
    std_f1 = file['F1'].std()
    std_metrics_avg = file['Metrics Average'].std()

    print(f"Best Accuracy: {best_accuracy}, Best Precision: {best_precision}, Best Recall: {best_recall}, Best F1: {best_f1}, Best Metrics Average: {best_metrics_avg}")
    print(f"Average Accuracy: {avg_accuracy}, Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1: {avg_f1}, Average Metrics Average: {avg_metrics_avg}")
    print(f"STD Accuracy: {std_accuracy}, STD Precision: {std_precision}, STD Recall: {std_recall}, STD F1: {std_f1}, STD Metrics Average: {std_metrics_avg}\n")

    top_accuracy = file.sort_values('Accuracy', ascending=False)
    top_precision = file.sort_values('Precision', ascending=False)
    top_recall = file.sort_values('Recall', ascending=False)
    top_f1 = file.sort_values('F1', ascending=False)
    metrics_average = file.sort_values(by='Metrics Average', ascending=False)

    columns = ['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1', 'Metrics Average']

    print(f"Top {num_rows} Accuracy:\n{top_accuracy[columns].head(num_rows)}")
    print(f"\nTop {num_rows} Precision:\n{top_precision[columns].head(num_rows)}")
    print(f"\nTop {num_rows} Recall:\n{top_recall[columns].head(num_rows)}")
    print(f"\nTop {num_rows} F1:\n{top_f1[columns].head(num_rows)}")
    print(f"\nTop {num_rows} Metrics Average:\n{metrics_average[columns].head(num_rows)}")

def optimize_hyperparameters(model_class, data, hidden_channels, space, csv_file, num_epochs=500, min_epochs=200, epoch_patience = 100, n_calls=50, random_state=0, print_epochs=True):
    """
    Optimize hyperparameters using Bayesian optimization.

    Args:
    model_class (type): The model class to be instantiated.
    data (torch_geometric.data.Data): The dataset containing features and edge indices.
    hidden_channels (int): The number of hidden channels in the model.
    space (list): The search space for hyperparameters.
    csv_file (str): The path to the CSV file to save results.
    num_epochs (int): The maximum number of epochs for training.
    min_epochs (int): The minimum number of epochs before early stopping can occur.
    epoch_patience (int): The patience for early stopping.
    n_calls (int): The number of calls to the objective function.
    random_state (int): The random state for reproducibility.
    print_epochs (bool): Whether to print epoch details.

    Returns:
    list: The best hyperparameters found during optimization.
    """
    def objective(params):
        param_dict = {
            'learning_rate': params[0],
            'weight_decay': params[1],
            'dropout_rate': params[2]
        }

        csv_columns = [
            'Model', 'Computation', 'Epochs', 'Hidden Channels', 'Learning Rate', 'Weight Decay',
            'Dropout Rate', 'Loss', 'Accuracy', 'Precision',
            'Recall', 'F1'
        ]

        write_header = not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0

        if write_header:
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(csv_columns)

        model = model_class(num_features=data.num_features, out_channels=hidden_channels, dropout_rate=param_dict['dropout_rate'])
        optimizer = Adam(model.parameters(), lr=param_dict['learning_rate'], weight_decay=param_dict['weight_decay'])
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        best_val_f1 = 0
        patience = epoch_patience
        patience_counter = 0
        real_epochs = 0

        epochs = range(num_epochs)
        epoch_iterator = tqdm(epochs, desc=f"Configuration {objective.run_count}/{n_calls} (LR:{params[0]}, WD:{params[1]}, DR:{params[2]})") if not print_epochs else epochs

        for epoch in epoch_iterator:
            real_epochs += 1
            train_loss = train(data, model, optimizer, criterion)
            accuracy, precision, recall, f1, _, _ = validate(data, model)

            if f1 > best_val_f1:
                best_val_f1 = f1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and epoch >= min_epochs:
                if print_epochs:
                    print(f"Early stopping at epoch {epoch}")
                else:
                    epoch_iterator.set_description(f"Configuration {objective.run_count}/{n_calls} (LR:{params[0]}, WD:{params[1]}, DR:{params[2]}) (Early stop)")
                break

            if print_epochs and epoch % 50 == 0:
                print(f"Configuration: {objective.run_count}/{n_calls}, Epoch: {epoch}, Loss: {train_loss:.4f}, Val Accuracy: {accuracy:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")

        end_time = time.time()
        computation = end_time - start_time
        computation = f'{computation // 60:.0f}m {computation % 60:.0f}s'

        if print_epochs:
            print(f"Successfully trained configuration {objective.run_count}/{n_calls}")

        with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([model_class.__name__, computation, real_epochs, hidden_channels, params[0], params[1], params[2], train_loss, accuracy, precision, recall, f1 ])

        return -best_val_f1
    
    objective.run_count = 0

    def wrapper(params):
        objective.run_count += 1
        if print_epochs:
            print(f"Training configuration {objective.run_count}/{n_calls} (LR:{params[0]}, WD:{params[1]}, DR:{params[2]})...")
        result = objective(params)
        return result

    res = gp_minimize(wrapper, space, n_calls=n_calls, random_state=random_state)
    best_hyperparams = res.x
    print("Best hyperparameters: ", best_hyperparams)
    return best_hyperparams

def evaluate_loss(data, model, criterion):
    """
    Evaluate the loss of the model on the given dataset.

    Args:
    data (torch_geometric.data.Data): The dataset containing features and edge indices.
    model (torch.nn.Module): The trained model to be evaluated.
    criterion (torch.nn.Module): The loss function.

    Returns:
    torch.Tensor: The computed loss.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        labels = data.y.to(device)
        loss = criterion(out, labels)
    return loss

def cross_validate(models_config, csv_file=None, num_epochs=500, n_splits=10, n_repeats=3, print_splits=True, early_stopping_patience=100, min_epochs=200):
    """
    Perform cross-validation for multiple model configurations.

    Args:
    models_config (list): A list of model configurations.
    csv_file (str): The path to the CSV file to save results.
    num_epochs (int): The maximum number of epochs for training.
    n_splits (int): The number of splits for cross-validation.
    n_repeats (int): The number of repeats for cross-validation.
    print_splits (bool): Whether to print split details.
    early_stopping_patience (int): The patience for early stopping.
    min_epochs (int): The minimum number of epochs before early stopping can occur.

    Returns:
    None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_count, config in enumerate(models_config, start=1):
        model_name = config['model_name']
        model_class = config['model']
        data_name = config['data_name']
        data = config['data']
        hidden_channels = config['hidden_channels']
        learning_rate = config['learning_rate']
        num_epochs = num_epochs
        weight_decay = config['weight_decay']
        dropout_rate = config['dropout_rate']
        criterion = config['criterion']

        print(f"\nTraining {model_name} model ({model_count}/{len(models_config)}) on {data_name} dataset...")

        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'loss': []
        }

        labels = data.y.cpu().numpy()

        total_splits = n_splits * n_repeats
        split_count = 1

        for repeat in range(n_repeats):
            if print_splits:
                split_iterator = enumerate(rskf.split(np.zeros(len(labels)), labels))
            else:
                tqdm_desc = f"Repetition {repeat + 1}/{n_repeats}"
                split_iterator = tqdm(enumerate(rskf.split(np.zeros(len(labels)), labels)), total=n_splits, desc=tqdm_desc)

            current_split = 0
            for fold, (train_index, test_index) in split_iterator:
                if current_split >= n_splits:
                    break

                train_mask = torch.zeros(data.num_nodes, dtype=bool)
                test_mask = torch.zeros(data.num_nodes, dtype=bool)
                train_mask[train_index] = True
                test_mask[test_index] = True

                data.train_mask = train_mask
                data.test_mask = test_mask

                model = model_class(num_features= data.num_features ,out_channels=hidden_channels, dropout_rate=dropout_rate).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                labels = data.y[data.train_mask]
                class_counts = labels.bincount()
                class_weights = 1. / class_counts.float()
                class_weights = class_weights / class_weights.sum()
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

                best_val_f1 = 0
                early_stopping_counter = 0

                for epoch in range(num_epochs):
                    loss = train(data, model, optimizer, criterion)
                    val_accuracy, val_precision, val_recall, val_f1, _, _ = validate(data, model)

                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        early_stopping_counter = 0
                    else:
                        early_stopping_counter += 1

                    if early_stopping_counter >= early_stopping_patience and epoch >= min_epochs:
                        if print_splits:
                            print(f'Early stopping at epoch {epoch+1}')
                        else:
                            split_iterator.set_description(tqdm_desc + '(Early stop)')
                        break

                    if (epoch + 1) % 50 == 0 and print_splits:
                        print(f"Repetition: {repeat + 1}/{n_repeats}, Split: {fold + 1}/{n_splits}, Epoch: {epoch + 1}/{num_epochs}, Val Loss: {loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

                accuracy, precision, recall, f1, _, _ = test(data, model)
                test_loss = evaluate_loss(data, model, criterion)

                results['accuracy'].append(accuracy)
                results['precision'].append(precision)
                results['recall'].append(recall)
                results['f1'].append(f1)
                results['loss'].append(test_loss)

                if print_splits:
                    print(f"Test Results: Split {fold + 1}/{n_splits}: Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

                current_split += 1
                split_count += 1

        metrics_summary = {
            'Metric': [],
            'Mean': [],
            'Standard Deviation': []
        }
        for key, value in results.items():
            metrics_summary['Metric'].append(key)
            metrics_summary['Mean'].append(np.mean(value))
            metrics_summary['Standard Deviation'].append(np.std(value))

        print(f"Successfully cross-validated {model_name} model ({model_count}/{len(models_config)}) on {data_name} dataset\n")
        for metric, mean, std in zip(metrics_summary['Metric'], metrics_summary['Mean'], metrics_summary['Standard Deviation']):
            print(f"{metric.capitalize()}: Mean: {mean:.4f}, Std: {std:.4f}")

        if csv_file:
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Metric', 'Mean', 'Standard Deviation'])
                for metric, mean, std in zip(metrics_summary['Metric'], metrics_summary['Mean'], metrics_summary['Standard Deviation']):
                    writer.writerow([metric.capitalize(), mean, std])

def misclassifications(data, model, metadata_save_path, save_path):
    """
    Identify and save misclassified nodes.

    Args:
    data (torch_geometric.data.Data): The dataset containing features, edge indices, and labels.
    model (torch.nn.Module): The trained model to be evaluated.
    metadata_save_path (str): The path to the metadata file containing node names.
    save_path (str): The path to the CSV file to save misclassified nodes.

    Returns:
    None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data = data.to(device)
    model.eval()
    
    print("\nFinding misclassified nodes...")
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out.argmax(dim=1)
        labels = data.y

    misclassified = preds != labels
    misclassified_indices = torch.where(misclassified)[0].cpu().numpy()

    with open(metadata_save_path, 'r') as f:
        metadata = json.load(f)
    idx_to_node = metadata['node_names']

    misclassified_nodes = [(idx_to_node[str(idx)], labels[idx].item(), preds[idx].item()) for idx in misclassified_indices]

    print(f"Sucessfully found {len(misclassified_nodes)} misclassified nodes and saved them into {save_path}")

    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Node', 'True Label', 'Predicted Label'])
        for node, true_label, pred_label in misclassified_nodes:
            writer.writerow([node, true_label, pred_label])

def datasets_evaluate(datasets, model, csv_file):
    """
    Evaluate a model on multiple datasets and save the results.

    Args:
    datasets (list): A list of dictionaries, each containing 'name' and 'data' keys.
    model (torch.nn.Module): The trained model to be evaluated.
    csv_file (str): The path to the CSV file to save results.

    Returns:
    None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"\nEvaluating {model.__class__.__name__} on different datasets...")
    
    results = []

    for dataset in datasets:
        name = dataset['name']
        data = dataset['data']

        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            
            test_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            
            probs = torch.softmax(out[test_mask], dim=1)[:, 1]
            preds = out[test_mask].argmax(dim=1)
            labels = data.y[test_mask].to(device)
            
            accuracy = accuracy_score(labels.cpu(), preds.cpu())
            precision = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
            recall = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
            f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)
            
            print(f'Dataset: {name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
            
            results.append([name, accuracy, precision, recall, f1])

    print(f"Successfully evaluated {model.__class__.__name__} on different datasets")

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
        writer.writerows(results)

def plot_feature_importance(metadata_path, model, data, num_features_to_plot=None, domain_names=None):
    """
    Plot feature importance for specified nodes or all nodes.

    Args:
    metadata_path (str): The path to the metadata file containing feature names and node names.
    model (torch.nn.Module): The trained model to be explained.
    data (torch_geometric.data.Data): The dataset containing features and edge indices.
    num_features_to_plot (int): The number of top features to plot.
    domain_names (list): The list of domain names for which to plot feature importance.

    Returns:
    list: A list of tuples containing the plotted figures and domain names.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    feature_names = metadata['feature_names']
    node_names = metadata['node_names']
    
    domain_to_index = {v: int(k) for k, v in node_names.items()}
    
    if domain_names is not None:
        node_indices = []
        for domain in domain_names:
            if domain in domain_to_index:
                node_indices.append(domain_to_index[domain])
            else:
                print(f"Warning: Domain name '{domain}' not found in metadata.")
    else:
        node_indices = np.arange(len(data.x))

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw'
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        model_config=model_config
    )

    figs = []

    if domain_names is None:
        explanation = explainer(data.x, data.edge_index, index=node_indices)
        feature_importance = explanation.node_mask.cpu().detach().numpy()
        mean_importance = feature_importance.mean(axis=0)

        if num_features_to_plot is None or num_features_to_plot > len(mean_importance):
            num_features_to_plot = len(mean_importance)

        indices = np.argsort(mean_importance)[-num_features_to_plot:][::-1]
        top_features = mean_importance[indices]
        top_feature_names = [feature_names[i] for i in indices]

        fig = plt.figure(figsize=(10, 7))
        plt.barh(range(num_features_to_plot), top_features, align='center')
        plt.yticks(range(num_features_to_plot), top_feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature Name')
        plt.title(f'Top {num_features_to_plot} Feature Importance for All Nodes')
        plt.gca().invert_yaxis()
        plt.show()

        figs.append((fig, "All Nodes"))
    else:
        for domain_name in domain_names:
            node_idx = domain_to_index[domain_name]
            explanation = explainer(data.x, data.edge_index, index=node_idx)
            feature_importance = explanation.node_mask.cpu().detach().numpy()
            mean_importance = feature_importance.mean(axis=0)

            if num_features_to_plot is None or num_features_to_plot > len(mean_importance):
                num_features_to_plot = len(mean_importance)

            indices = np.argsort(mean_importance)[-num_features_to_plot:][::-1]
            top_features = mean_importance[indices]
            top_feature_names = [feature_names[i] for i in indices]

            fig = plt.figure(figsize=(10, 7))
            plt.barh(range(num_features_to_plot), top_features, align='center')
            plt.yticks(range(num_features_to_plot), top_feature_names)
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Name')
            plt.title(f'Top {num_features_to_plot} Feature Importance for Node \"{domain_name}\" ({model.__class__.__name__})')
            plt.gca().invert_yaxis()
            plt.show()

            figs.append((fig, domain_name))

    return figs


def plot_explain_subgraphs(model, data, domain_names, metadata_path, misclassification_csv, num_epochs=200):
    """
    Plot explanation subgraphs for specified domain names.

    Args:
    model (torch.nn.Module): The trained model to be explained.
    data (torch_geometric.data.Data): The dataset containing features, labels and edge indices.
    domain_names (list): The list of domain names for which to plot subgraphs.
    metadata_path (str): The path to the metadata file containing node names.
    misclassification_csv (str): The path to the CSV file containing misclassification data.
    num_epochs (int): The number of epochs for the explainer optimization algorithm
    
    Returns:
    list: A list of tuples containing the plotted figures and domain names.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    node_names = metadata['node_names']
    
    reverse_node_names = {v: k for k, v in node_names.items()}
    
    # Read the misclassification CSV
    misclassification_df = pd.read_csv(misclassification_csv)
    misclassified_domains = set(misclassification_df['Node'].tolist())

    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw'
    )

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=num_epochs),
        explanation_type='model',
        edge_mask_type='object',
        model_config=model_config
    )

    def visualize_custom_graph(edge_index, edge_weight, node_labels, title, data_labels, focus_node):
        g = nx.DiGraph()
        node_size = 800

        for node in edge_index.view(-1).unique().tolist():
            g.add_node(node_labels[node])

        for (src, dst), w in zip(edge_index.t().tolist(), edge_weight.tolist()):
            src = node_labels[src]
            dst = node_labels[dst]
            g.add_edge(src, dst, alpha=w)

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(g)
        for src, dst, data in g.edges(data=True):
            ax.annotate(
                '',
                xy=pos[src],
                xytext=pos[dst],
                arrowprops=dict(
                    arrowstyle="->",
                    alpha=data['alpha'],
                    shrinkA=sqrt(node_size) / 2.0,
                    shrinkB=sqrt(node_size) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ),
            )

        node_colors = ['white' for _ in g.nodes()]
        node_edgecolors = ['red' if data_labels[node] == 1 else 'blue' for node in g.nodes()]
        node_edgewidths = [2 if node == node_labels[focus_node] else 1 for node in g.nodes()]

        nx.draw(g, pos, with_labels=False, node_size=node_size, node_color=node_colors, edgecolors=node_edgecolors, linewidths=node_edgewidths, ax=ax)

        for node, (x, y) in pos.items():
            fontweight = 'bold' if node == node_labels[focus_node] else 'normal'
            fontcolor = 'red' if node in misclassified_domains else 'green'
            plt.text(x, y, s=node, fontsize=10, fontdict={'color': fontcolor, 'weight': fontweight}, horizontalalignment='center', verticalalignment='center')

        # Add legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Tracker', markersize=10, markerfacecolor='white', markeredgecolor='red')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Non-Tracker', markersize=10, markerfacecolor='white', markeredgecolor='blue')
        plt.legend(handles=[red_patch, blue_patch])

        plt.title(title)
        plt.show()
        
        return fig

    figs = []
    for domain_name in domain_names:
        if domain_name not in reverse_node_names:
            print(f"Domain name {domain_name} not found in node names.")
            continue
        
        node_index = int(reverse_node_names[domain_name])
        explanation = explainer(data.x, data.edge_index, index=node_index)
        subgraph = explanation.get_explanation_subgraph()
        edge_index = subgraph.edge_index
        edge_weight = subgraph.edge_mask
        subgraph_node_indices = set(edge_index[0].tolist() + edge_index[1].tolist())

        if not subgraph_node_indices:
            print(f"No nodes found in the subgraph for domain '{domain_name}'.")
            continue

        node_labels = {idx: node_names.get(str(idx), str(idx)) for idx in subgraph_node_indices}

        # Mapping node indices to their labels in data.y
        data_labels = {node_labels[idx]: data.y[idx].item() for idx in subgraph_node_indices}

        title = f"Explanation Subgraph For Node \"{domain_name}\" ({model.__class__.__name__})"
        fig = visualize_custom_graph(edge_index, edge_weight, node_labels, title, data_labels, node_index)
        figs.append((fig, domain_name))
    
    return figs

def print_node_names(metadata_path, indices):
    """
    Print node names for given indices based on metadata.

    Args:
    metadata_path (str): The path to the metadata file containing node names.
    indices (list): The list of node indices to print names for.

    Returns:
    None
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    node_names = metadata['node_names']
    
    for number in indices:
        if str(number) in node_names:
            print(f'{number}: "{node_names[str(number)]}"')
        else:
            print(f'{number}: "Not found"')
