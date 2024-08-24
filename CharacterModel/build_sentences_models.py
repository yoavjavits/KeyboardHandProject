from train import build_model
from generate_sentences import generate_text
import os


def run_model_creation(models_folder_, model_num_, embed_size, num_heads, num_layers, num_hidden, dropout, bptt, epochs=None):
    folder_name = 'model_' + str(model_num_)
    model_folder = os.path.join(models_folder_, folder_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_name = os.path.join(model_folder, 'model.pt')
    logfile_name = os.path.join(model_folder, 'log.txt')
    generate_text_name = os.path.join(model_folder, 'generated_text.txt')
    model_params_text_name = os.path.join(model_folder, 'model_params.txt')

    args_dict = {'emsize': embed_size, 'nhead': num_heads, 'nlayers': num_layers, 'nhid': num_hidden, 'dropout': dropout, 'bptt': bptt,
                 'model_name': model_name, 'logfile_name': logfile_name,
                 'data': './sentences'}
    if epochs is not None:
        args_dict['epochs'] = epochs

    build_model(args_dict)
    generate_text({'checkpoint': model_name, 'outf': generate_text_name})

    with open(model_params_text_name, 'w') as f:
        f.write('Embed size: {}\n'.format(embed_size))
        f.write('Number of heads: {}\n'.format(num_heads))
        f.write('Number of layers: {}\n'.format(num_layers))
        f.write('Number of hidden units: {}\n'.format(num_hidden))
        f.write('Dropout: {}\n'.format(dropout))
        f.write('BPTT: {}\n'.format(bptt))


if __name__ == '__main__':
    # models folder
    models_folder = os.path.join(os.getcwd(), 'models')
    models_folder = os.path.join(models_folder, 'sentences_only')

    # first model - default
    print('-' * 50)
    print('Creating model 1')

    model_num = 1
    embed_size = 512
    num_heads = 8
    num_layers = 8
    num_hidden = 1024
    dropout = 0.2
    bptt = 100
    run_model_creation(models_folder, model_num, embed_size, num_heads, num_layers, num_hidden, dropout, bptt)

    # second model - larger
    print('-' * 50)
    print('Creating model 2')

    model_num = 2
    embed_size = 512
    num_heads = 8
    num_layers = 12
    num_hidden = 2048
    dropout = 0.1
    bptt = 100
    run_model_creation(models_folder, model_num, embed_size, num_heads, num_layers, num_hidden, dropout, bptt)

    # third model - smaller
    print('-' * 50)
    print('Creating model 3')

    model_num = 3
    embed_size = 256
    num_heads = 4
    num_layers = 4
    num_hidden = 1024
    dropout = 0.2
    bptt = 100
    run_model_creation(models_folder, model_num, embed_size, num_heads, num_layers, num_hidden, dropout, bptt)

    # fourth model - default with larger BPTT
    print('-' * 50)
    print('Creating model 4')

    model_num = 4
    embed_size = 512
    num_heads = 8
    num_layers = 8
    num_hidden = 1024
    dropout = 0.1
    bptt = 150
    run_model_creation(models_folder, model_num, embed_size, num_heads, num_layers, num_hidden, dropout, bptt)

    # fifth model - extra large
    print('-' * 50)
    print('Creating model 5')

    model_num = 5
    embed_size = 1024
    num_heads = 8
    num_layers = 16
    num_hidden = 2048
    dropout = 0.2
    bptt = 150
    epochs = 50
    run_model_creation(models_folder, model_num, embed_size, num_heads, num_layers, num_hidden, dropout, bptt, epochs)
