import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from model_v1 import TypingLSTMV1
from model_v2 import TypingLSTMV2, FocalLoss
import pickle
from torch.utils.data import DataLoader
from HandPosDataset import HandPosDataset
from sklearn.metrics import roc_auc_score


def parse_args(parser):
    parser.add_argument('--model_name', type=str, default='./models/third_version/demo/model.pt')
    parser.add_argument('--model_type', type=str, default='many')
    parser.add_argument('--model_version', type=int, default=1)

    parser.add_argument('--data_dir', type=str, default='./data/regular')
    parser.add_argument('--log_file', type=str, default='./models/third_version/demo/log.txt')

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('-loss', type=str, default='focal')
    parser.add_argument('--alpha', type=float, default=0.82)
    parser.add_argument('--gamma', type=float, default=8)

    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def train(model, dataloader, optimizer, criterion, epoch, device, model_type, threshold):
    model.train()

    total_loss = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    count_positive_samples = 0

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        # Move data to device
        input_ids = input_ids.type(torch.float32).to(device)
        targets = targets.type(torch.float32).to(device)

        # Forward pass
        outputs = model(input_ids)

        # Calculate loss
        if model_type == 'many':
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, targets)

        elif model_type == 'one':
            outputs = outputs.squeeze(-1)
            outputs = outputs[:, -1]
            loss = criterion(outputs, targets)

        else:
            raise ValueError(f'Invalid model type: {model_type}')

        # Metrics calculation
        predictions = (torch.sigmoid(outputs) > threshold).float()

        targets_ = targets.view(-1)
        predictions_ = predictions.view(-1)

        true_positives += ((predictions_ == 1) & (targets_ == 1)).sum().item()
        true_negatives += ((predictions_ == 0) & (targets_ == 0)).sum().item()
        false_positives += ((predictions_ == 1) & (targets_ == 0)).sum().item()
        false_negatives += ((predictions_ == 0) & (targets_ == 1)).sum().item()

        count_positive_samples += (targets_ == 1).sum().item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logging
        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / (batch_idx + 1)

            accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

            print(f'Epoch [{epoch}], Step [{batch_idx +1}/{len(dataloader)}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}', flush=True)
            print(f'Number of positive samples: {count_positive_samples}, Number of positive labeled: {true_positives + false_positives}', flush=True)

    avg_epoch_loss = total_loss / len(dataloader)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print(f'Training Epoch {epoch} - Average loss: {avg_epoch_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}', flush=True)
    print(f'Number of positive samples: {count_positive_samples}, Number of positive labeled: {true_positives + false_positives}', flush=True)
    return {'loss': avg_epoch_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall}


def evaluate(model, dataloader, criterion, device, model_type, threshold):
    model.eval()

    total_loss = 0

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    targets_total = None
    probabilities_total = None

    count_positive_samples = 0

    with torch.no_grad():
        for input_ids, targets in dataloader:

            # Move data to device
            input_ids = input_ids.type(torch.float32).to(device)
            targets = targets.type(torch.float32).to(device)

            # Forward pass
            outputs = model(input_ids)

            # Calculate loss
            if model_type == 'many':
                outputs = outputs.squeeze(-1)
                loss = criterion(outputs, targets)

            elif model_type == 'one':
                outputs = outputs.squeeze(-1)
                outputs = outputs[:, -1]
                loss = criterion(outputs, targets)

            else:
                raise ValueError(f'Invalid model type: {model_type}')

            total_loss += loss.item()

            # Metrics calculation
            predictions = (torch.sigmoid(outputs) > threshold).float()

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

    avg_loss = total_loss / len(dataloader)
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    auc = roc_auc_score(targets_total, probabilities_total)

    return avg_loss, accuracy, precision, recall, auc, count_positive_samples, true_positives + false_positives


def build_lstm(args_dict=None):
    parser = argparse.ArgumentParser()
    args = parse_args(parser)

    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_version == 1:
        model = TypingLSTMV1(input_size=6, hidden_size=args.hidden_size, num_layers=args.num_layers, model_type=args.model_type)
        model.to(device)

    elif args.model_version == 2:
        model = TypingLSTMV2(input_size=6, hidden_size=args.hidden_size, num_layers=args.num_layers, model_type=args.model_type)
        model.to(device)

    else:
        raise ValueError(f'Invalid model version: {args.model_version}')

    print('Loading data...', flush=True)

    with open(args.data_dir + '/train.pkl', 'rb') as f:
        train_dataset: HandPosDataset = pickle.load(f)
    with open(args.data_dir + '/valid.pkl', 'rb') as f:
        valid_dataset: HandPosDataset = pickle.load(f)
    with open(args.data_dir + '/test.pkl', 'rb') as f:
        test_dataset: HandPosDataset = pickle.load(f)

    train_dataset.update_seq_length(args.seq_len)
    valid_dataset.update_seq_length(args.seq_len)
    test_dataset.update_seq_length(args.seq_len)

    train_dataset.update_dataset_type(args.model_type)
    valid_dataset.update_dataset_type(args.model_type)
    test_dataset.update_dataset_type(args.model_type)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print('Data loaded!', flush=True)

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)
    else:
        raise ValueError(f'Invalid loss function: {args.loss}')

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    log_file = open(args.log_file, 'w')

    def print_current_line(line):
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    best_valid_auc = 0
    last_seq = 0

    print("Start training", flush=True)
    for epoch in range(1, args.epochs + 1):
        if last_seq > 10:
            print("Early stopping")
            break

        print(f'\n===== Epoch {epoch} =====')

        # Training
        _ = train(model, train_loader, optimizer, criterion, epoch, device, args.model_type, args.threshold)

        # Validation
        valid_loss, valid_accuracy, valid_precision, valid_recall, auc,\
            count_positive_samples, positive_labeled_samples = (
            evaluate(model, valid_loader, criterion, device, args.model_type, args.threshold))
        print_current_line(f'Epoch {epoch} - Average loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}, Precision: {valid_precision:.4f}, Recall: {valid_recall:.4f}, AUC: {auc:.4f},'
                           f' Number of positive samples: {count_positive_samples}, Number of positive labeled: {positive_labeled_samples}')

        # Save the model if validation loss has decreased
        if auc > best_valid_auc:
            best_valid_auc = auc

            with open(args.model_name, 'wb') as f:
                torch.save(model, f)

            print('Model saved!\n', flush=True)

            last_seq = 0

        else:
            last_seq += 1

    print("Training finished", flush=True)

    # Test
    # Load the best model
    with open(args.model_name, 'rb') as f:
        model = torch.load(f)
        model = model.to(device)

    # Evaluate on test set
    test_loss, test_accuracy, test_precision, test_recall, auc, test_positive_sample, test_positive_labeled = evaluate(model, test_loader, criterion, device, args.model_type, args.threshold)
    print_current_line(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, AUC: {auc:.4f}, Number of positive samples: {test_positive_sample}, Number of positive labeled: {test_positive_labeled}')

    # close the log file
    log_file.close()


if __name__ == '__main__':
    build_lstm()
