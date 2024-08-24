import torch
import argparse
from data_scripts.data_sentences import Corpus as CorpusSentences
from data_scripts.data_reformer import Corpus as CorpusReformer
import os
import matplotlib.pyplot as plt


def create_args(parser):
    parser.add_argument('--model', type=str, default='./model.pt', help='model checkpoint to use')
    parser.add_argument('--model_type', type=str, default='reformer', help='type of model to use')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', default=True, action='store_true', help='use CUDA')

    parser.add_argument('--data_base_path', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--data', type=str, default='./sentences', help='data corpus to use')

    args = parser.parse_args()

    return args


def extract_sentences(corpus_):
    eos_id = corpus_.dictionary.char2idx['eos']

    sentences_ = []

    curr_index = 0
    while curr_index < len(corpus_.test):
        sentence = []
        while curr_index < len(corpus_.test) and corpus_.test[curr_index] != eos_id:
            sentence.append(corpus_.test[curr_index])
            curr_index += 1

        sentences_.append(sentence)
        curr_index += 1

    return sentences_


def create_letters_count(sentence_, corpus_):
    dict_letter = {}
    for char_idx in sentence_:
        char = corpus_.dictionary.idx2char[char_idx]

        if char not in dict_letter:
            dict_letter[char] = {'count': 1, 'correct': 0}

        else:
            dict_letter[char]['count'] += 1

    for char in range(ord('a'), ord('z') + 1):
        char = chr(char)
        if char not in dict_letter:
            dict_letter[char] = {'count': 0, 'correct': 0}

    if ' ' not in dict_letter:
        dict_letter[' '] = {'count': 0, 'correct': 0}

    return dict_letter


def get_word_lengths(sentence_, backspace_id):
    word_lengths = []

    curr_index = 0
    while curr_index < len(sentence_):
        word = []
        while curr_index < len(sentence_) and sentence_[curr_index] != backspace_id:
            word.append(sentence_[curr_index])
            curr_index += 1

        word_lengths.append(len(word))
        curr_index += 1

    return word_lengths


def calculate_statistics(sentence_, restored_sentence, corpus_):
    backspace_id = corpus_.dictionary.char2idx[' ']

    num_correct = 0
    num_all = len(sentence_)

    num_letters_correct = 0
    num_letters_all = len([c for c in sentence_ if c != backspace_id])

    num_backspace_correct = 0
    num_backspace_all = num_all - num_letters_all

    word_lengths = get_word_lengths(sentence_, backspace_id)

    num_first_half_word_correct = 0
    num_first_half_word_all = sum([(c / 2) for c in word_lengths])

    num_second_half_word_correct = 0
    num_second_half_word_all = sum(word_lengths) - num_first_half_word_all

    num_first_half_sentence_correct = 0
    num_first_half_sentence_all = len(sentence_) // 2

    num_second_half_sentence_correct = 0
    num_second_half_sentence_all = len(sentence_) - num_first_half_sentence_all

    letters_count = create_letters_count(sentence_, corpus_)

    curr_word_index = 0
    curr_word_letter_pos = 0
    for i in range(len(sentence_)):
        if sentence_[i] == restored_sentence[i]:
            num_correct += 1

            if i < len(sentence_) // 2:
                num_first_half_sentence_correct += 1
            else:
                num_second_half_sentence_correct += 1

            if sentence_[i] != backspace_id:
                num_letters_correct += 1

                curr_word_length = word_lengths[curr_word_index]
                if curr_word_letter_pos < curr_word_length // 2:
                    num_first_half_word_correct += 1
                else:
                    num_second_half_word_correct += 1

            else:
                num_backspace_correct += 1

            letters_count[corpus_.dictionary.idx2char[sentence_[i]]]['correct'] += 1

        if sentence_[i] == backspace_id:
            curr_word_index += 1
            curr_word_letter_pos = 0
        else:
            curr_word_letter_pos += 1

    letters_accuracy = (num_letters_correct / num_letters_all) if num_letters_all != 0 else 0
    backspace_accuracy = (num_backspace_correct / num_backspace_all) if num_backspace_all != 0 else 0
    first_half_word_accuracy = (num_first_half_word_correct / num_first_half_word_all) if num_first_half_word_all != 0 else 0
    second_half_word_accuracy = (num_second_half_word_correct / num_second_half_word_all) if num_second_half_word_all != 0 else 0
    first_half_sentence_accuracy = (num_first_half_sentence_correct / num_first_half_sentence_all) if num_first_half_sentence_all != 0 else 0
    second_half_sentence_accuracy = (num_second_half_sentence_correct / num_second_half_sentence_all) if num_second_half_sentence_all != 0 else 0

    return {'num_correct': num_correct, 'num_all': num_all, 'accuracy': num_correct / num_all,
            'num_letters_correct': num_letters_correct, 'num_letters_all': num_letters_all,
            'letters_accuracy': letters_accuracy,
            'num_backspace_correct': num_backspace_correct, 'num_backspace_all': num_backspace_all,
            'backspace_accuracy': backspace_accuracy,
            'num_first_half_word_correct': num_first_half_word_correct,
            'num_first_half_word_all': num_first_half_word_all,
            'first_half_word_accuracy': first_half_word_accuracy,
            'num_second_half_word_correct': num_second_half_word_correct,
            'num_second_half_word_all': num_second_half_word_all,
            'second_half_word_accuracy': second_half_word_accuracy,
            'num_first_half_sentence_correct': num_first_half_sentence_correct,
            'num_first_half_sentence_all': num_first_half_sentence_all,
            'first_half_sentence_accuracy': first_half_sentence_accuracy,
            'num_second_half_sentence_correct': num_second_half_sentence_correct,
            'num_second_half_sentence_all': num_second_half_sentence_all,
            'second_half_sentence_accuracy': second_half_sentence_accuracy,
            'letters_count': letters_count}


def evaluate_sentence(sentence_, model_, corpus_, args_, device_):
    input_ids = sentence_[0].view(-1, 1).to(device_)
    print('Evaluate sentence')
    print(f'Start with inputs ids of shape: {input_ids.shape}')

    restored_sentence = [sentence_[0].item()]

    for i in range(1, len(sentence_)):
        output = model_(input_ids)

        if args_.model_type == 'reformer':
            output = output['logits']

        _, predicted = torch.max(output, -1)
        predicted = predicted.view(-1)[-1]

        restored_sentence.append(predicted.item())

        next_char = sentence_[i].view(-1,1).to(device_)
        print('Sentence next char shape:', next_char.shape)

        input_ids = torch.cat((input_ids, next_char), dim=1)
        print('Input ids shape:', input_ids.shape)

    sentence_ = [c.item() for c in sentence_]
    return_dict = calculate_statistics(sentence_, restored_sentence, corpus_)
    return_dict['original_sentence'] = sentence_
    return_dict['restored_sentence'] = restored_sentence

    return return_dict


def evaluate_sentences(sentences_, model_, corpus_, args_, device_, output_file=None):
    def print_line(line):
        print(line)
        if output_file is not None:
            output_file.write(line + '\n')

    statistics = {
        'num_all': 0,
        'num_correct': 0,
        'num_letters_all': 0,
        'num_letters_correct': 0,
        'num_backspace_all': 0,
        'num_backspace_correct': 0,
        'num_first_half_word_all': 0,
        'num_first_half_word_correct': 0,
        'num_second_half_word_all': 0,
        'num_second_half_word_correct': 0,
        'num_first_half_sentence_all': 0,
        'num_first_half_sentence_correct': 0,
        'num_second_half_sentence_all': 0,
        'num_second_half_sentence_correct': 0,
        'letters_count': {char: {'count': 0, 'correct': 0} for char in [chr(x) for x in range(ord('a'), ord('z') + 1)] + [' ']}
    }

    sentence_index = 0
    for sentence in sentences_:
        result = evaluate_sentence(sentence, model_, corpus_, args_, device_)

        print_line('Original: ' + ''.join([corpus_.dictionary.idx2char[x] for x in result['original_sentence']]))
        print_line('Restored: ' + ''.join([corpus_.dictionary.idx2char[x] for x in result['restored_sentence']]))
        print_line(f'Accuracy: {result["accuracy"]:.4f} | Letters accuracy: {result["letters_accuracy"]:.4f} | Backspace accuracy: {result["backspace_accuracy"]:.4f} | First half word accuracy: {result["first_half_word_accuracy"]:.4f} | Second half word accuracy: {result["second_half_word_accuracy"]:.4f}\n')

        statistics['num_all'] += result['num_all']
        statistics['num_correct'] += result['num_correct']
        statistics['num_letters_all'] += result['num_letters_all']
        statistics['num_letters_correct'] += result['num_letters_correct']
        statistics['num_backspace_all'] += result['num_backspace_all']
        statistics['num_backspace_correct'] += result['num_backspace_correct']
        statistics['num_first_half_word_all'] += result['num_first_half_word_all']
        statistics['num_first_half_word_correct'] += result['num_first_half_word_correct']
        statistics['num_second_half_word_all'] += result['num_second_half_word_all']
        statistics['num_second_half_word_correct'] += result['num_second_half_word_correct']
        statistics['num_first_half_sentence_all'] += result['num_first_half_sentence_all']
        statistics['num_first_half_sentence_correct'] += result['num_first_half_sentence_correct']
        statistics['num_second_half_sentence_all'] += result['num_second_half_sentence_all']
        statistics['num_second_half_sentence_correct'] += result['num_second_half_sentence_correct']
        for char in result['letters_count']:
            statistics['letters_count'][char]['count'] += result['letters_count'][char]['count']
            statistics['letters_count'][char]['correct'] += result['letters_count'][char]['correct']

        if sentence_index % 100 == 0:
            print('-' * 50)
            print(f'Processed {sentence_index} sentences')
            print('-' * 50)

        sentence_index += 1

    statistics['accuracy'] = statistics['num_correct'] / statistics['num_all']
    statistics['letters_accuracy'] = statistics['num_letters_correct'] / statistics['num_letters_all']
    statistics['backspace_accuracy'] = statistics['num_backspace_correct'] / statistics['num_backspace_all']
    statistics['first_half_word_accuracy'] = statistics['num_first_half_word_correct'] / statistics['num_first_half_word_all']
    statistics['second_half_word_accuracy'] = statistics['num_second_half_word_correct'] / statistics['num_second_half_word_all']
    statistics['first_half_sentence_accuracy'] = statistics['num_first_half_sentence_correct'] / statistics['num_first_half_sentence_all']
    statistics['second_half_sentence_accuracy'] = statistics['num_second_half_sentence_correct'] / statistics['num_second_half_sentence_all']
    for char in statistics['letters_count']:
        if statistics['letters_count'][char]['count'] != 0:
            statistics['letters_count'][char]['accuracy'] = statistics['letters_count'][char]['correct'] / statistics['letters_count'][char]['count']

        else:
            statistics['letters_count'][char]['accuracy'] = 0

    return statistics


def print_statistics(stats, output_file=None):
    def print_line(line):
        print(line)
        if output_file is not None:
            output_file.write(line + '\n')

    print_line(f'Accuracy: {stats["accuracy"]:.4f}')
    print_line(f'Letters accuracy: {stats["letters_accuracy"]:.4f}')
    print_line(f'Backspace accuracy: {stats["backspace_accuracy"]:.4f}')
    print_line(f'First half word accuracy: {stats["first_half_word_accuracy"]:.4f}')
    print_line(f'Second half word accuracy: {stats["second_half_word_accuracy"]:.4f}')
    print_line('Letters accuracy per letter:')

    chars_stats = ''
    for char in stats['letters_count']:
        if stats['letters_count'][char]['count'] != 0:
            chars_stats += f'{char}: {stats["letters_count"][char]["accuracy"]:.4f} '

    print_line(chars_stats)


def plot_letters_hist(stats, output_file=None):
    letters = [chr(x) for x in range(ord('a'), ord('z') + 1)] + [' ']
    accuracies = [stats['letters_count'][char]['accuracy'] for char in letters]

    plt.figure(figsize=(10, 5))
    plt.bar(letters, accuracies)
    plt.xlabel('Letter')
    plt.ylabel('Accuracy')
    plt.title('Letters accuracy')

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()


def plot_accuracy_hist(stats, output_file=None):
    accuracies = [stats['accuracy'], stats['letters_accuracy'], stats['backspace_accuracy'], stats['first_half_word_accuracy'], stats['second_half_word_accuracy'],
                  stats['first_half_sentence_accuracy'], stats['second_half_sentence_accuracy']]
    names = ['All', 'Letters', 'Space', 'First half \nword', 'Second half \nword', 'First half \nsentence', 'Second half \nsentence']

    plt.figure(figsize=(15, 10))
    plt.bar(names, accuracies)
    plt.xlabel('Type')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()


def evaluate_model(args_dict=None):
    parser = argparse.ArgumentParser(description='PyTorch Character Transformer Model')
    args = create_args(parser)

    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    if args.cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    with open(args.model, 'rb') as f:
        print('Loading model from', args.model)
        model = torch.load(f, map_location=device)

    model.eval()

    data_path = os.path.join(args.data_base_path, args.data)

    if args.model_type == 'sentences':
        print('Using sentences model')
        corpus = CorpusSentences(data_path)

    elif args.model_type == 'reformer':
        print('Using reformer model')
        corpus = CorpusReformer(data_path)

    else:
        raise ValueError('Data type not supported')

    sentences = extract_sentences(corpus)

    # get the current model folder
    output_folder = os.path.join(os.path.dirname(args.model), 'evaluation')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    restored_sentences = os.path.join(output_folder, 'restored_sentences.txt')
    final_statistics = os.path.join(output_folder, 'final_statistics.txt')
    letters_hist = os.path.join(output_folder, 'letters_hist.png')
    accuracy_hist = os.path.join(output_folder, 'accuracy_hist.png')

    with open(restored_sentences, 'w') as f:
        stats = evaluate_sentences(sentences, model, corpus, args, device, output_file=f)

    with open(final_statistics, 'w') as f:
        print_statistics(stats, output_file=f)

    plot_letters_hist(stats, output_file=letters_hist)

    plot_accuracy_hist(stats, output_file=accuracy_hist)


if __name__ == '__main__':
    evaluate_model({'model': './models/reformer/model_1/model.pt', 'model_type': 'reformer'})

    # evaluate_model({'model': './models/text8_only/model_2/model.pt', 'model_type': 'sentences'})
    # evaluate_model({'model': './models/text8_only/model_1/model.pt', 'model_type': 'sentences'})

    # evaluate_model({'model': './models/sentences_only/model_1/model.pt', 'model_type': 'sentences'})
    # evaluate_model({'model': './models/sentences_only/model_2/model.pt', 'model_type': 'sentences'})
    # evaluate_model({'model': './models/sentences_only/model_3/model.pt', 'model_type': 'sentences'})
    # evaluate_model({'model': './models/sentences_only/model_4/model.pt', 'model_type': 'sentences'})
    # evaluate_model({'model': './models/sentences_only/model_5/model.pt', 'model_type': 'sentences'})
