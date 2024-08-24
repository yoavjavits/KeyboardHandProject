import os


def get_model_params(model_folder):
    model_params_file_path = model_folder + '/model_params.txt'

    if not os.path.exists(model_params_file_path):
        return None

    model_params_lines = open(model_params_file_path, 'r').read().split('\n')
    model_type = model_params_lines[0].split(' ')[-1]
    model_seq_len = int(model_params_lines[4].split(' ')[-1])
    model_arch = 1 if model_seq_len == 3 else 2 if model_seq_len == 5 else 3
    model_alpha = float(model_params_lines[8].split(' ')[-1])
    model_gamma = float(model_params_lines[9].split(' ')[-1])
    batch_size = int(model_params_lines[2].split(' ')[-1])

    return {'type': model_type, 'arch': model_arch, 'alpha': model_alpha, 'gamma': model_gamma, 'folder': model_folder,
            'seq_len': model_seq_len, 'batch_size': batch_size}


def get_model_auc(model_folder):
    model_auc_file_path = model_folder + '/evaluation/results.txt'

    if not os.path.exists(model_auc_file_path):
        return None

    return float(open(model_auc_file_path, 'r').read().split('\n')[3].split(' ')[-1])


def get_model_precision(model_folder):
    model_precision_file_path = model_folder + '/evaluation/results.txt'

    if not os.path.exists(model_precision_file_path):
        return None

    return float(open(model_precision_file_path, 'r').read().split('\n')[1].split(' ')[-1])


def get_model_recall(model_folder):
    model_recall_file_path = model_folder + '/evaluation/results.txt'

    if not os.path.exists(model_recall_file_path):
        return None

    return float(open(model_recall_file_path, 'r').read().split('\n')[2].split(' ')[-1])


def get_model_f1(model_folder):
    model_f1_file_path = model_folder + '/evaluation/results.txt'

    if not os.path.exists(model_f1_file_path):
        return None

    precision = get_model_precision(model_folder)
    recall = get_model_recall(model_folder)

    return 2 * (precision * recall) / (precision + recall)
