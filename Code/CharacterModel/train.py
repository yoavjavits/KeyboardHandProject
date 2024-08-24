import argparse
import time
import torch
import torch.nn as nn
import torch.onnx
import os

from data_scripts.data_sentences import Corpus as CorpusSentences
from data_scripts.data_wiki_chars import Corpus as CorpusWikiChars
from data_scripts.data_wiki_words import Corpus as CorpusWikiWords
from data_scripts.data_text8 import Corpus as CorpusText8
from model import TransformerModel


def parse_args(parser):
    parser.add_argument('--emsize', type=int, default=512, help='size of character embeddings')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--nlayers', type=int, default=12, help='number of layers')
    parser.add_argument('--nhid', type=int, default=1024, help='number of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')

    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size')
    parser.add_argument('--bptt', type=int, default=150, help='sequence length')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='report interval')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N', help='evaluation batch size')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')

    parser.add_argument('--data_base_path', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--data', type=str, default='./sentences', help='location of the data corpus')
    parser.add_argument('--chars', type=bool, default=True, help='train on characters')

    parser.add_argument('--model_name', type=str, default='model.pt', help='path to save the final model')
    parser.add_argument('--logfile-name', type=str, default='log_file.txt', help='path to save the log file')

    args_ = parser.parse_args()

    return args_


def batchify(data_, bsz, device_):
    """
    :param device_: 
    :param data_: tensor of shape [N, ] where N is the number of characters in the file, including the eos token
    :param bsz: batch size
    :return: tensor of shape [number of batches, batch size], where each row is a sequence of length bsz

    Docs:
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e.g. 'g' on 'f' can't be learned, but allows more efficient
    batch processing.
    """

    # Work out how cleanly we can divide the dataset into bsz parts.
    num_batch = data_.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_ = data_.narrow(0, 0, num_batch * bsz)
    # Evenly divide the data across the bsz batches.
    data_ = data_.view(bsz, -1).t().contiguous()

    return data_.to(device_)


def get_batch(source, i, args_):
    """
    :param source: tensor of shape [number of batches, batch size]
    :param i: index of the batch

    :return: Data, target, where data has shape [sequence length, batch size] and target has shape [sequence length * batch size]

    Docs:
    get_batch subdivides the source data into chunks of length args.bptt.
    If the source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivision of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension.
    """
    seq_len = min(args_.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def train(epoch_, criterion_, optimizer_, model_, train_data_, args_):
    """
    Train the model for one epoch
    :param epoch_: epoch number
    :param criterion_: loss function
    :param optimizer_: optimizer
    :param model_: model to train
    :param train_data_: training data
    :param args_: arguments
    """
    model_.train()

    total_loss = 0.
    total_correct_predictions = 0.
    total_tokens = 0
    start_time = time.time()

    for batch, i in enumerate(range(0, train_data_.size(0) - 1, args_.bptt)):
        data, targets = get_batch(train_data_, i, args_)
        model_.zero_grad()
        output = (model_(data))
        output = output.view(-1, model_.num_tokens)

        loss = criterion_(output, targets)
        loss.backward()

        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        correct_predictions = (predicted == targets).sum().item()
        total_correct_predictions += correct_predictions

        # Update total tokens
        total_tokens += len(targets)

        optimizer_.step()

        total_loss += loss.item()

        if batch % args_.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args_.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | accuracy {:5.2f}'.
                  format(epoch_, batch, len(train_data_) // args_.bptt,
                         elapsed * 1000 / args_.log_interval, cur_loss, total_correct_predictions / total_tokens))

            total_loss = 0.
            total_correct_predictions = 0.
            total_tokens = 0

            start_time = time.time()


def evaluate(model_, criterion_, eval_data, args_):
    """
    Evaluate the model on the evaluation data, calculating average loss, accuracy, perplexity, and CER.

    Args:
    - model_ (torch.nn.Module): The model to evaluate.
    - criterion_ (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
    - eval_data (torch.Tensor): The evaluation dataset.
    - args_ (Namespace): Arguments containing batch size and sequence length (bptt).

    Returns:
    - A dictionary containing average loss, accuracy, perplexity, and CER.
    """
    model_.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    total_tokens = 0
    total_batches = 0
    total_correct_predictions = 0
    batch_loss_sum = 0

    with torch.no_grad():  # Disable gradient calculation
        for i in range(0, eval_data.size(0) - 1, args_.bptt):
            data, targets = get_batch(eval_data, i, args_)
            output = model_(data).view(-1, model_.num_tokens)

            # Calculate loss
            loss = criterion_(output, targets)
            total_loss += loss.item() * len(targets)
            batch_loss_sum += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions = (predicted == targets).sum().item()
            total_correct_predictions += correct_predictions

            # Update total tokens
            total_tokens += len(targets)

            total_batches += 1

    average_loss = batch_loss_sum / total_batches  # The loss is averaged over all mini-batches
    accuracy = total_correct_predictions / total_tokens
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()  # Calculate perplexity

    return {
            'average_loss': average_loss,
            'accuracy': accuracy,
            'perplexity': perplexity
    }


def build_model(args_dict=None):
    # Parse arguments
    args = parse_args(argparse.ArgumentParser())
    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    if args.cuda:
        print('Using CUDA')
        device = torch.device("cuda")
    else:
        print('Using CPU')
        device = torch.device("cpu")

    # load data
    data_path = os.path.join(args.data_base_path, args.data)

    if args.data == './sentences':
        print('Using sentences data')
        corpus = CorpusSentences(data_path)

    elif args.data == './wiki':
        if args.chars:
            print('Using wiki data with characters')
            corpus = CorpusWikiChars(data_path)
        else:
            print('Using wiki data with words')
            corpus = CorpusWikiWords(data_path)

    elif args.data == './text8':
        print('Using text8 data')
        corpus = CorpusText8(data_path)

    else:
        raise ValueError('Invalid data path')

    train_data = batchify(corpus.train, args.batch_size, device)
    val_data = batchify(corpus.valid, args.eval_batch_size, device)
    test_data = batchify(corpus.test, args.eval_batch_size, device)

    # build the model
    num_tokens = len(corpus.dictionary)
    model = TransformerModel(num_tokens=num_tokens,
                             embed_dim=args.emsize,
                             num_heads=args.nhead,
                             num_layers=args.nlayers,
                             num_hidden=args.nhid,
                             dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Training
    best_val_loss = None
    lines_to_log = []

    def print_current_line(line):
        print(line)
        lines_to_log.append(line)

    try:
        print_current_line('Starting training...')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            train(epoch, criterion, optimizer, model, train_data, args)
            eval_dict = evaluate(model, criterion, val_data, args)

            print_current_line('-' * 90)
            print_current_line('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid accuracy {:5.2f} '
                               '| valid perplexity {:5.2f}'.
                               format(epoch, (time.time() - epoch_start_time), eval_dict['average_loss'],
                                      eval_dict['accuracy'], eval_dict['perplexity']))
            print_current_line('-' * 90)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or eval_dict['average_loss'] < best_val_loss:
                with open(args.model_name, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = eval_dict['average_loss']

    except KeyboardInterrupt:
        print_current_line('-' * 90)
        print_current_line('Exiting from training early')

    # load the best saved model.
    with open(args.model_name, 'rb') as f:
        model = torch.load(f)
        model = model.to(device)

    # Run on test data.
    test_dict = evaluate(model, criterion, test_data, args)
    print_current_line('=' * 90)
    print_current_line('| End of training | test loss {:5.2f} | test accuracy {:5.2f} | test perplexity {:5.2f} '
                       .format(test_dict['average_loss'], test_dict['accuracy'], test_dict['perplexity']))

    print_current_line('=' * 90)

    # Save the log file
    with open(args.logfile_name, 'w') as f:
        for line in lines_to_log:
            f.write(line + '\n')


if __name__ == '__main__':
    build_model({'emsize': 1024,
                 'nhead': 8,
                 'nlayers': 16,
                 'nhid': 2048,
                 'dropout': 0.3,
                 'data': './text8'
})
