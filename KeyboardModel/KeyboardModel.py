import torch.nn as nn
import torch
from charmodel import TransformerModel
from model_v2 import TypingLSTMV2
import TypingModelSelection
import os


def chose_best_typing_model(models_dir, criteria):
    models = [m for m in os.listdir(models_dir) if m.startswith('model')]

    models_params = [TypingModelSelection.get_model_params(os.path.join(models_dir, m)) for m in models]

    # Remove model with missing params
    models_params = [m for m in models_params if m is not None]

    # Get sorting function
    sort_function = None
    if criteria == 'auc':
        sort_function = TypingModelSelection.get_model_auc
    elif criteria == 'precision':
        sort_function = TypingModelSelection.get_model_precision
    elif criteria == 'recall':
        sort_function = TypingModelSelection.get_model_recall
    elif criteria == 'f1':
        sort_function = TypingModelSelection.get_model_f1
    else:
        raise ValueError('Invalid typing model criteria')

    # Remove models without results
    models_params = [m for m in models_params if sort_function(m['folder']) is not None]

    # Get best model
    best_model = sorted(models_params, key=lambda x: sort_function(x['folder']), reverse=True)[0]
    best_model_path = str(best_model['folder']) + '/model.pt'

    return best_model_path, best_model['seq_len']


class KeyboardModel(nn.Module):
    def __init__(self, char_model_path='../CharacterModel/models/reformer/model_1/model.pt',
                 typing_models_dir='../TypingModel/models/',
                 typing_models_version=2,
                 typing_model_criteria='auc',
                 device='cpu'):

        super(KeyboardModel, self).__init__()

        self.char_model: TransformerModel = torch.load(char_model_path)
        self.char_model.to(device)

        if typing_model_criteria not in ['auc', 'precision', 'recall', 'f1']:
            raise ValueError('Invalid typing model criteria')

        if typing_models_version not in [1, 2, 3]:
            raise ValueError('Invalid typing model version')

        if typing_models_version == 1:
            typing_models_dir = typing_models_dir + 'first_version/'
        elif typing_models_version == 2:
            typing_models_dir = typing_models_dir + 'second_version/'
        elif typing_models_version == 3:
            typing_models_dir = typing_models_dir + 'third_version/'
        else:
            raise ValueError('Invalid typing model version')

        best_model_path, typing_seq_len = chose_best_typing_model(typing_models_dir, typing_model_criteria)
        self.typing_model: TypingLSTMV2 = torch.load(best_model_path)
        self.typing_model.to(device)

        self.typing_seq_len = typing_seq_len

        self.char_model.eval()
        self.typing_model.eval()




