import os
from train import build_lstm
from evalutae_model import evaluate_model
import argparse
from HandPosDataset import HandPosDataset


def run_model_creation(models_folder_, model_type_, model_num_, epochs_, batch_size_, lr_, seq_len_, hidden_size_, num_layers_, threshold_):
    folder_name = f'model_{model_type}_{model_num_}'
    model_folder = os.path.join(models_folder_, folder_name)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    model_name = os.path.join(model_folder, 'model.pt')
    logfile_name = os.path.join(model_folder, 'log.txt')
    model_params_text_name = os.path.join(model_folder, 'model_params.txt')
    model_evaluation_output_folder = model_folder + '/evaluation'

    print('Building model', flush=True)
    args_dict_build = {
        'model_name': model_name,
        'model_type': model_type_,
        'model_version': 1,
        'log_file': logfile_name,
        'epochs': epochs_,
        'batch_size': batch_size_,
        'lr': lr_,
        'seq_len': seq_len_,
        'hidden_size': hidden_size_,
        'num_layers': num_layers_,
        'threshold': threshold_,
        'loss': 'bce',
        'data_dir': './data/regular'
    }
    build_lstm(args_dict_build)

    print('Evaluating model', flush=True)
    args_dict_eval = {
        'model': model_name,
        'model_type': model_type_,
        'batch_size': batch_size_,
        'seq_len': seq_len_,
        'threshold': threshold_,
        'output_folder': model_evaluation_output_folder,
        'data': './data/regular',
    }
    evaluate_model(args_dict_eval)

    with open(model_params_text_name, 'w') as f:
        f.write(f'Model type {model_type_}\n')
        f.write(f'Epochs {epochs_}\n')
        f.write(f'Batch size {batch_size_}\n')
        f.write(f'Learning rate {lr_}\n')
        f.write(f'Sequence length {seq_len_}\n')
        f.write(f'Hidden size {hidden_size_}\n')
        f.write(f'Number of layers {num_layers_}\n')
        f.write(f'Threshold {threshold_}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build models')
    parser.add_argument('--model_num', type=int, default=4, help='Model number')

    args = parser.parse_args()

    # models folder
    models_folder = os.path.join(os.getcwd(), 'models/first_version')

    print(f'Creating models {args.model_num}', flush=True)

    # models 1 - model-type many
    if args.model_num == -1 or args.model_num == 1:
        print('-' * 50, flush=True)
        print('Creating model 1 of type many', flush=True)

        model_type = 'many'
        model_num = 1
        epochs = 50
        batch_size = 64
        lr = 0.001
        seq_len = 30
        hidden_size = 256
        num_layers = 4
        threshold = 0.5

        run_model_creation(models_folder, model_type, model_num, epochs, batch_size, lr, seq_len, hidden_size, num_layers, threshold)

    # models 2 - model-type many
    if args.model_num == -1 or args.model_num == 2:
        print('-' * 50, flush=True)
        print('Creating model 2 of type many', flush=True)

        model_type = 'many'
        model_num = 2
        epochs = 50
        batch_size = 64
        lr = 0.0001
        seq_len = 50
        hidden_size = 512
        num_layers = 8
        threshold = 0.5

        run_model_creation(models_folder, model_type, model_num, epochs, batch_size, lr, seq_len, hidden_size, num_layers, threshold)

    # models 3 - model-type one
    if args.model_num == -1 or args.model_num == 3:
        print('-' * 50, flush=True)
        print('Creating model 3 of type one', flush=True)

        model_type = 'one'
        model_num = 1
        epochs = 50
        batch_size = 64
        lr = 0.001
        seq_len = 30
        hidden_size = 256
        num_layers = 4
        threshold = 0.5

        run_model_creation(models_folder, model_type, model_num, epochs, batch_size, lr, seq_len, hidden_size, num_layers, threshold)

    # models 4 - model-type one
    if args.model_num == -1 or args.model_num == 4:
        print('-' * 50, flush=True)
        print('Creating model 4 of type one', flush=True)

        model_type = 'one'
        model_num = 2
        epochs = 50
        batch_size = 64
        lr = 0.0001
        seq_len = 50
        hidden_size = 512
        num_layers = 8
        threshold = 0.5

        run_model_creation(models_folder, model_type, model_num, epochs, batch_size, lr, seq_len, hidden_size, num_layers, threshold)

    print('Done', flush=True)
