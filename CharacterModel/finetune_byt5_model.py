import torch
import torch.nn as nn
from transformers import T5EncoderModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import argparse


def parse_args(parser):
    parser.add_argument('--model_name', type=str, default='./models/byt5/model.pt',
                        help='path to save the final model')
    parser.add_argument('--logfile-name', type=str, default='./models/byt5/log_file.txt',
                        help='path to save the log file')

    return parser.parse_args()


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        """
        file_path: Path to the text file (e.g., 'train.txt')
        tokenizer: The ByT5 tokenizer
        block_size: The length of the input sequences
        """
        self.tokenizer = tokenizer
        self.block_size = block_size

        with open(file_path, 'r', encoding="utf8") as f:
            text = f.read()

        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.squeeze(0)
        self.data = tokens

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Input sequence
        x = self.data[idx:idx + self.block_size]
        # Target is the next character in the sequence
        y = self.data[idx + self.block_size]
        return x, y


class ByT5ForNextCharPrediction(nn.Module):
    def __init__(self):
        super(ByT5ForNextCharPrediction, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained('google/byt5-base')
        self.head = nn.Linear(self.encoder.config.d_model, self.encoder.config.vocab_size)  # Linear layer on top

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        # We only care about the output of the last token in the sequence
        last_hidden_state = sequence_output[:, -1, :]  # [batch_size, hidden_dim]
        logits = self.head(last_hidden_state)  # [batch_size, vocab_size]
        return logits


def train(model_, dataloader, optimizer_, criterion_, device_, epoch):
    model_.train()

    total_loss = 0
    total_correct = 0
    total_predictions = 0

    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device_)
        targets = targets.to(device_)

        optimizer_.zero_grad()
        logits = model_(input_ids)

        # Compute the loss
        loss = criterion_(logits, targets)
        loss.backward()
        optimizer_.step()

        total_loss += loss.item()

        # Compute the accuracy
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).sum().item()
        total_correct += correct
        total_predictions += len(targets.view(-1))

        # Logging
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(
                f'Epoch [{epoch}], Step [{batch_idx + 1}/{len(dataloader)}], Loss: {avg_loss:.4f}, Accuracy: {total_correct / total_predictions:.4f}', flush=True)

    avg_epoch_loss = total_loss / len(dataloader)
    print(f'====> Epoch: {epoch} Average loss: {avg_epoch_loss:.4f}', flush=True)
    return avg_epoch_loss, total_correct / total_predictions


def evaluate(model_, dataloader, criterion_, device_):
    model_.eval()

    total_loss = 0
    total_correct = 0
    total_predictions = 0

    with torch.no_grad():
        for input_ids, targets in dataloader:
            # Move data to device
            input_ids = input_ids.to(device_)
            targets = targets.to(device_)

            # Forward pass
            outputs = model_(input_ids)

            loss = criterion_(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item()

            # Accuracy
            _, predictions = torch.max(outputs.logits, -1)
            total_correct += (predictions == targets).sum().item()
            total_predictions += len(targets.view(-1))

    avg_loss = total_loss / len(dataloader)
    print(f'====> Evaluation Average loss: {avg_loss:.4f}, Accuracy: {total_correct / total_predictions:.4f}', flush=True)
    return avg_loss, total_correct / total_predictions


if __name__ == '__main__':
    print('Start fine-tune byt5 encoder model', flush=True)

    args = parse_args(argparse.ArgumentParser())

    block_size = 128
    batch_size = 32
    learning_rate = 0.001
    epochs = 10

    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')

    # Load datasets
    train_dataset = TextDataset('./data/sentences/train.txt', tokenizer, block_size)
    valid_dataset = TextDataset('./data/sentences/valid.txt', tokenizer, block_size)
    test_dataset = TextDataset('./data/sentences/test.txt', tokenizer, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = ByT5ForNextCharPrediction()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    f = open(args.logfile_name, 'w')

    def print_current_line(line):
        print(line, flush=True)
        f.write(line + '\n')


    # Training loop
    best_valid_loss = float('inf')
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}", flush=True)
        train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device, epoch)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion, device)
        print_current_line(
            f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy} | "
            f"Validation Loss: {valid_loss:.4f} | Validation Accuracy: {valid_accuracy}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            with open(args.model_name, 'wb') as f:
                torch.save(model, f)

    print("Training completed", flush=True)

    # Test the model
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print_current_line(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy}")
