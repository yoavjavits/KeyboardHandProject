import torch
from transformers import ReformerModelWithLMHead, ReformerConfig
from torch.utils.data import Dataset, DataLoader
from data_scripts.data_reformer import Corpus
import argparse


def parse_args(parser):
    parser.add_argument('--model_name', type=str, default='./models/reformer/model.pt', help='path to save the final model')
    parser.add_argument('--logfile-name', type=str, default='./models/reformer/log_file.txt', help='path to save the log file')
    
    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y


def train(model, dataloader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_predictions = 0

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        # Move data to device
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))

        # Accuracy
        _, predictions = torch.max(outputs.logits, -1)
        total_correct += (predictions == targets).sum().item()
        total_predictions += len(targets.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Logging
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f'Epoch [{epoch}], Step [{batch_idx +1}/{len(dataloader)}], Loss: {avg_loss:.4f}, Accuracy: {total_correct / total_predictions:.4f}', flush=True)

    avg_epoch_loss = total_loss / len(dataloader)
    print(f'====> Epoch: {epoch} Average loss: {avg_epoch_loss:.4f}', flush=True)
    return avg_epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for input_ids, targets in dataloader:

            # Move data to device
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(input_ids)

            loss = criterion(outputs.logits.view(-1, outputs.logits.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Accuracy
            _, predictions = torch.max(outputs.logits, -1)
            total_correct += (predictions == targets).sum().item()
            total_predictions += len(targets.view(-1))

    avg_loss = total_loss / len(dataloader)
    print(f'====> Evaluation Average loss: {avg_loss:.4f}, Accuracy: {total_correct / total_predictions:.4f}', flush=True)
    return avg_loss, total_correct / total_predictions


if __name__ == '__main__':
    print('Start fine-tune the Reformer model', flush=True)

    args = parse_args(argparse.ArgumentParser())

    corpus: Corpus = Corpus("./data/sentences")

    block_size = 128  # Sequence length to consider for training

    # load datasets and dataloaders
    train_dataset = TextDataset(corpus.train, block_size)
    valid_dataset = TextDataset(corpus.valid, block_size)
    test_dataset = TextDataset(corpus.test, block_size)

    train_loader = DataLoader(train_dataset, batch_size=64)
    valid_loader = DataLoader(valid_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = ReformerConfig.from_pretrained("google/reformer-enwik8")
    conf.axial_pos_embds = False
    conf.max_position_embeddings = 512
    print("Model configuration loaded", flush=True)
    model = ReformerModelWithLMHead.from_pretrained("google/reformer-enwik8", config=conf)
    print("Model loaded", flush=True)
    model = model.to(device)
    print("Model moved to device", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = torch.nn.CrossEntropyLoss()

    lines_to_log = []

    def print_current_line(line):
        print(line, flush=True)
        lines_to_log.append(line)

    # Training and evaluation functions
    num_epochs = 30  # Set the number of epochs
    best_valid_loss = float('inf')

    print_current_line("Start training")
    for epoch in range(1, num_epochs + 1):
        print_current_line(f'\n===== Epoch {epoch} =====')

        # Training
        train_loss = train(model, train_loader, optimizer, criterion, epoch, device)

        # Validation
        valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device)
        print_current_line(f'Evaluation Epoch {epoch} ====> Evaluation Average loss: {valid_loss:.4f}, Accuracy: {valid_accuracy:.4f}')

        # Save the model if validation loss has decreased
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            with open(args.model_name, 'wb') as f:
                torch.save(model, f)

            print('Model saved!\n', flush=True)

    print("Training finished", flush=True)

    # Test
    # Load the best model
    with open(args.model_name, 'rb') as f:
        model = torch.load(f)
        model = model.to(device)

    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print_current_line(f'\nTest Loss: {test_loss:.4f}')

    # Save the log file
    with open(args.logfile_name, 'w') as f:
        for line in lines_to_log:
            f.write(line + '\n')