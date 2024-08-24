import os
from train import build_lstm
from evalutae_model import evaluate_model
from HandPosDataset import HandPosDataset


def run_model_creation(models_folder_, model_type_, model_num_, epochs_, batch_size_, lr_, seq_len_, hidden_size_, num_layers_, threshold_,
                       alpha_, gamma_):
    folder_name = f'model_{model_type_}_{model_num_}'
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
        'model_version': 2,
        'log_file': logfile_name,
        'epochs': epochs_,
        'batch_size': batch_size_,
        'lr': lr_,
        'seq_len': seq_len_,
        'hidden_size': hidden_size_,
        'num_layers': num_layers_,
        'threshold': threshold_,
        'loss': 'focal',
        'alpha': alpha_,
        'gamma': gamma_,
        'data_dir': './data/filtered'
    }
    build_lstm(args_dict_build)

    print('Evaluating model', flush=True)
    args_dict_eval = {
        'model': model_name,
        'model_type': model_type_,
        'model_version': 2,
        'batch_size': batch_size_,
        'seq_len': seq_len_,
        'threshold': threshold_,
        'output_folder': model_evaluation_output_folder,
        'data': './data/filtered'
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
        f.write(f'Alpha {alpha_}\n')
        f.write(f'Gamma {gamma_}\n')


if __name__ == '__main__':
    # models folder
    models_folder = os.path.join(os.getcwd(), 'models/second_version')

    # first_arch = {'epochs': 100, 'batch_size': 32, 'lr': 0.001, 'seq_len': 3, 'hidden_size': 128, 'num_layers': 2}
    # second_arch = {'epochs': 100, 'batch_size': 48, 'lr': 0.001, 'seq_len': 5, 'hidden_size': 256, 'num_layers': 4}
    # third_arch = {'epochs': 100, 'batch_size': 64, 'lr': 0.001, 'seq_len': 7, 'hidden_size': 512, 'num_layers': 8}
    #
    # alpha_params = [0.5, 0.75, 0.78, 0.8, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89, 0.9]
    # gamma_params = [2, 4, 6, 8, 10]
    #
    # total_models = len(alpha_params) * len(gamma_params) * 3
    # count_model = 0
    #
    # print('Total models', total_models, flush=True)
    #
    # for alpha in alpha_params:
    #     for gamma in gamma_params:
    #         for arch in [first_arch, second_arch, third_arch]:
    #             count_model += 1
    #             print(f'Creating model {count_model}', flush=True)
    #             run_model_creation(models_folder, 'many', count_model, arch['epochs'], arch['batch_size'], arch['lr'], arch['seq_len'], arch['hidden_size'], arch['num_layers'], 0.5, alpha, gamma)
    #             run_model_creation(models_folder, 'one', count_model, arch['epochs'], arch['batch_size'], arch['lr'], arch['seq_len'], arch['hidden_size'], arch['num_layers'], 0.5, alpha, gamma)
    #
    # print('Done', flush=True)

    first_arch = {'epochs': 100, 'batch_size': 32, 'lr': 0.001, 'seq_len': 3, 'hidden_size': 128, 'num_layers': 2}
    second_arch = {'epochs': 100, 'batch_size': 48, 'lr': 0.001, 'seq_len': 5, 'hidden_size': 256, 'num_layers': 4}
    third_arch = {'epochs': 100, 'batch_size': 64, 'lr': 0.001, 'seq_len': 7, 'hidden_size': 512, 'num_layers': 8}

    alpha_params = [0.86, 0.87, 0.88, 0.89, 0.9]
    gamma_params = [4, 6, 8, 10]

    total_models = len(alpha_params) * len(gamma_params) * 3
    count_model = 13

    print('Total models', total_models, flush=True)

    for arch in [first_arch, second_arch, third_arch]:
        for alpha in alpha_params:
            for gamma in gamma_params:
                    count_model += 1
                    print(f'Creating model {count_model}', flush=True)
                    run_model_creation(models_folder, 'many', count_model, arch['epochs'], arch['batch_size'], arch['lr'], arch['seq_len'], arch['hidden_size'], arch['num_layers'], 0.5, alpha, gamma)
                    run_model_creation(models_folder, 'one', count_model, arch['epochs'], arch['batch_size'], arch['lr'], arch['seq_len'], arch['hidden_size'], arch['num_layers'], 0.5, alpha, gamma)

    print('Done', flush=True)
