import argparse
import torch

from data_scripts.data_sentences import Corpus as CorpusSentences
from data_scripts.data_wiki_chars import Corpus as CorpusWikiChars
from data_scripts.data_wiki_words import Corpus as CorpusWikiWords
from data_scripts.data_text8 import Corpus as CorpusText8


def create_args(parser):
    parser.add_argument('--data', type=str, default='./sentences', help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model.pt', help='model checkpoint to use')
    parser.add_argument('--outf', type=str, default='generated.txt', help='output file for generated text')
    parser.add_argument('--chars', type=int, default='1000', help='number of chars to generate')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--cuda', default=True, action='store_true', help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature - higher will increase diversity')
    parser.add_argument('--log-interval', type=int, default=50, help='reporting interval')
    parser.add_argument('--max-seq-length', type=int, default=512, help='maximum sequence length')
    parser.add_argument('--wiki_chars', default=False, help='use wiki chars')
    parser.add_argument('--wiki_words', default=False, help="use wiki words")

    args = parser.parse_args()

    return args


def generate_text(args_dict=None):
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

    if args.temperature < 1e-3:
        parser.error("--temperature has to be greater or equal 1e-3.")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    if args.data == './text8':
        print('Loading Corpus Text8')
        corpus = CorpusText8(args.data)

    elif args.data == './wiki':
        if args.wiki_chars:
            print('Load Corpus wiki chars')
            corpus = CorpusWikiChars(args.data)
        elif args.wiki_words:
            print('Loading Corpus wiki words')
            corpus = CorpusWikiWords(args.data)
        else:
            raise ValueError('Unknown wiki data type')

    elif args.data == './sentences':
        print('Loading Corpus Sentences')
        corpus = CorpusSentences(args.data)

    else:
        raise ValueError('Unknown data type')

    num_tokens = len(corpus.dictionary)

    input_ = torch.randint(num_tokens, (1, 1), dtype=torch.long).to(device)

    with open(args.outf, 'w') as out_f:
        with torch.no_grad():  # no tracking history
            for i in range(args.chars):
                output = model(input_, False)
                char_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                char_idx = torch.multinomial(char_weights, 1)[0]
                char_tensor = torch.Tensor([[char_idx]]).long().to(device)

                # append the new character
                input_ = torch.cat([input_, char_tensor], 0)

                # check the maximum sequence length
                if input_.size(0) >= args.max_seq_length:
                    input_ = input_[1:]

                if args.wiki_words:
                    word = corpus.dictionary.idx2word[char_idx]
                    out_f.write(word + ('\n' if i % 20 == 19 else ' '))

                else:
                    char = corpus.dictionary.idx2char[char_idx.item()]
                    out_f.write(char if char != 'eos' else '\n')

                if i % args.log_interval == 0:
                    print('| Generated {}/{} chars'.format(i, args.chars))


if __name__ == '__main__':
    generate_text({'data': './text8',
                   'checkpoint': './model.pt'})
