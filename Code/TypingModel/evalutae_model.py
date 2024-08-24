import argparse
import torch
import pickle
from torch.utils.data import DataLoader
from HandPosDataset import HandPosDataset
from model_v1 import TypingLSTMV1
from model_v2 import TypingLSTMV2
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import os


def create_args(parser):
    parser.add_argument('--model', type=str, default='./models/third_version/demo/model.pt', help='path to model')
    parser.add_argument('--model_version', type=int, default=2, help='model version')
    parser.add_argument('--model_type', type=str, default='many', help='model type')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--sequence_length', type=int, default=30, help='sequence length')
    parser.add_argument('--threshold', type=float, default=0.5, help='threshold for binary classification')

    parser.add_argument('--data', type=str, default='./data/regular', help='path to data')
    parser.add_argument('--output_folder', type=str, default='./models/third_version/demo/evaluation', help='path to output folder')

    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args


def print_results_to_text(results, output_folder):
    with open(output_folder + '/results.txt', 'w') as f:
        f.write(f'Accuracy: {results["accuracy"]}\n')
        f.write(f'Precision: {results["precision"]}\n')
        f.write(f'Recall: {results["recall"]}\n')
        f.write(f'AUC: {results["auc"]}\n')
        f.write(f'Count of ground truth positive samples: {results["count_gt_positive"]}\n')
        f.write(f'Count of predicted positive samples: {results["count_predicted_positive"]}\n')


def plot_bar_chart(results, output_folder):
    # first plot
    labels = ['Accuracy', 'Precision', 'Recall', 'AUC']
    values = [results['accuracy'], results['precision'], results['recall'], results['auc']]

    plt.bar(labels, values)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Metrics values')
    plt.savefig(output_folder + '/results_chart.png')
    plt.close()

    # second plot
    labels = ['Count of ground truth \n positive samples', 'Count of predicted \n positive samples']
    values = [results['count_gt_positive'], results['count_predicted_positive']]

    plt.bar(labels, values)
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Positive samples count')
    plt.savefig(output_folder + '/positive_samples_results_chart.png')
    plt.close()


def evaluate_model(args_dict=None):
    args = create_args(argparse.ArgumentParser())

    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model_version == 1:
        model: TypingLSTMV1 = torch.load(args.model)
        model.to(device)
    elif args.model_version == 2:
        model: TypingLSTMV2 = torch.load(args.model)
        model.to(device)
    else:
        raise ValueError(f'Invalid model version: {args.model_version}')

    with open(args.data + '/test.pkl', 'rb') as f:
        test_data: HandPosDataset = pickle.load(f)

    test_data.update_seq_length(args.sequence_length)
    test_data.update_dataset_type(args.model_type)

    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model.eval()

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    targets_total = None
    probabilities_total = None

    count_positive_samples = 0

    with torch.no_grad():
        for input_ids, targets in test_loader:

            # Move data to devicedef
            input_ids = input_ids.type(torch.float32).to(device)
            targets = targets.type(torch.float32).to(device)

            # Forward pass
            outputs = model(input_ids)

            # Calculate loss
            if model.model_type == 'many':
                outputs = outputs.squeeze(-1)

            elif model.model_type == 'one':
                outputs = outputs.squeeze(-1)
                outputs = outputs[:, -1]

            else:
                raise ValueError(f'Invalid model type: {model.model_type}')

            #  Metrics calculation
            predictions = (torch.sigmoid(outputs) > args.threshold).float()

            targets_ = targets.view(-1)
            predictions_ = predictions.view(-1)

            true_positives += ((predictions_ == 1) & (targets_ == 1)).sum().item()
            true_negatives += ((predictions_ == 0) & (targets_ == 0)).sum().item()
            false_positives += ((predictions_ == 1) & (targets_ == 0)).sum().item()
            false_negatives += ((predictions_ == 0) & (targets_ == 1)).sum().item()

            count_positive_samples += (targets_ == 1).sum().item()

            # For AUC calculation
            if targets_total is None:
                targets_total = targets.view(-1).cpu().numpy()
            else:
                targets_total = np.concatenate([targets_total, targets.view(-1).cpu().numpy()])

            if probabilities_total is None:
                probabilities_total = torch.sigmoid(outputs).view(-1).cpu().detach().numpy()
            else:
                probabilities_total = np.concatenate([probabilities_total, torch.sigmoid(outputs).view(-1).cpu().detach().numpy()])

    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    auc = roc_auc_score(targets_total, probabilities_total)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'count_gt_positive': count_positive_samples,
        'count_predicted_positive': true_positives + false_positives
    }

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    print_results_to_text(results, args.output_folder)
    plot_bar_chart(results, args.output_folder)


if __name__ == '__main__':
    evaluate_model()
