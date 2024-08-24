import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from KeyboardModel import KeyboardModel
import argparse


def parse_args(parser):
    parser.add_argument('--model_criteria', type=str, default='auc', help='Model criteria to evaluate')

    parser.add_argument('--threshold', type=float, default=0.46, help='Threshold for binary classification')

    parser.add_argument('--data_dir', type=str, default='../../Data', help='Directory with data')
    parser.add_argument('--output_file', type=str, default='./results/results.txt', help='Output file with results')

    return parser.parse_args()


def get_restored_sentence(df_):
    df_tmp = df_.copy()
    # restore typed sentence
    df_tmp['keys_prev'] = df_tmp['keys'].shift(1).bfill()
    pressed_keys_vectors = df_tmp.apply(lambda x: x['keys_prev'] if not np.array_equal(x['keys_prev'], x['keys']) else np.nan, axis=1).dropna() # get rows where pressing state was changed
    pressed_keys_indices = pressed_keys_vectors.apply(lambda x: np.argmax(x) if x.sum() > 0 else -1) # extract indices from one-hot vectors (pressing states)
    pressed_keys_asciis = pressed_keys_indices.values[np.where(pressed_keys_indices.values >= 0)[0]] + 65 # extract ascii chars
    pressed_keys_asciis[np.where(pressed_keys_asciis == 91)[0]] = 32 # fix space ascii char
    restored_sentence = ''.join(chr(i) for i in pressed_keys_asciis) # convert array of ascii chars to visible text
    restored_sentence = restored_sentence.lower()

    return restored_sentence


def get_minmax_values(df_):
    df_tmp = df_.copy()

    minmax_values = {'thumb3_rel': {'min': [np.inf, np.inf, np.inf], 'max': [-np.inf, -np.inf, -np.inf]},
                     'thumb4_rel': {'min': [np.inf, np.inf, np.inf], 'max': [-np.inf, -np.inf, -np.inf]}}

    # Get the relative positions
    df_tmp['thumb3_rel'] = df_tmp['thumb3'] - df_tmp['dev_pos']
    df_tmp['thumb4_rel'] = df_tmp['thumb4'] - df_tmp['dev_pos']

    # Create separation between x, y and z values
    for pos in range(3):
        df_tmp[f'thumb3_rel_{pos}'] = df_tmp['thumb3_rel'].apply(lambda x: x[pos])
        df_tmp[f'thumb4_rel_{pos}'] = df_tmp['thumb4_rel'].apply(lambda x: x[pos])

    # Calculate the min and max values
    for pos in range(3):
        minmax_values['thumb3_rel']['min'][pos] = min(minmax_values['thumb3_rel']['min'][pos], min(df_tmp[f'thumb3_rel_{pos}']))
        minmax_values['thumb3_rel']['max'][pos] = max(minmax_values['thumb3_rel']['max'][pos], max(df_tmp[f'thumb3_rel_{pos}']))

        minmax_values['thumb4_rel']['min'][pos] = min(minmax_values['thumb4_rel']['min'][pos], min(df_tmp[f'thumb4_rel_{pos}']))
        minmax_values['thumb4_rel']['max'][pos] = max(minmax_values['thumb4_rel']['max'][pos], max(df_tmp[f'thumb4_rel_{pos}']))

    return minmax_values

def calculate_avg_distance_diff_for_pressed_seq(df_):
    distance_diff_dict = {'thumb3_rel': [], 'thumb4_rel': []}

    curr_row = 0
    while curr_row < len(df_):
        if df_['press'].iloc[curr_row] == 1:
            start_point_thumb3_rel = np.array(df_['thumb3_rel'].iloc[curr_row])
            start_point_thumb4_rel = np.array(df_['thumb4_rel'].iloc[curr_row])

            while curr_row < len(df_) and df_['press'].iloc[curr_row] == 1:
                curr_row += 1

            end_point_thumb3_rel = np.array(df_['thumb3_rel'].iloc[curr_row - 1])
            end_point_thumb4_rel = np.array(df_['thumb4_rel'].iloc[curr_row - 1])

            distance_diff_dict['thumb3_rel'].append(np.linalg.norm(end_point_thumb3_rel - start_point_thumb3_rel))
            distance_diff_dict['thumb4_rel'].append(np.linalg.norm(end_point_thumb4_rel - start_point_thumb4_rel))

        else:
            curr_row += 1

    distance_diff_dict['thumb3_rel'] = np.mean(distance_diff_dict['thumb3_rel'])
    distance_diff_dict['thumb4_rel'] = np.mean(distance_diff_dict['thumb4_rel'])

    return distance_diff_dict


def is_rows_close(row1, row2, distance_diff_dict, threshold=0.5):
    thumb3_diff = np.linalg.norm(np.array(row1['thumb3_rel']) - np.array(row2['thumb3_rel']))
    thumb4_diff = np.linalg.norm(np.array(row1['thumb4_rel']) - np.array(row2['thumb4_rel']))

    if thumb3_diff < distance_diff_dict['thumb3_rel'] * threshold or thumb4_diff < distance_diff_dict['thumb4_rel'] * threshold:
        return True

    return False


def create_filtered_df(df_):
    avg_diff_dist_for_pressed_seq = calculate_avg_distance_diff_for_pressed_seq(df_)

    rows = []

    curr_row = 0
    while curr_row < len(df_):
        rows.append(df_.iloc[curr_row])

        next_row = curr_row + 1
        while next_row < len(df_) and is_rows_close(df_.iloc[curr_row], df_.iloc[next_row],  avg_diff_dist_for_pressed_seq):
            next_row += 1

        curr_row = next_row

    return pd.DataFrame(rows)


def normalize_and_filter_df(df_, to_filter=False):
    minmax_values = get_minmax_values(df_)

    # Get the relative positions
    df_['thumb3_rel'] = df_['thumb3'] - df_['dev_pos']
    df_['thumb4_rel'] = df_['thumb4'] - df_['dev_pos']

    # Normalize the relative positions
    df_['thumb3_rel'] = df_['thumb3_rel'].apply(
        lambda x: [(x[i] - minmax_values['thumb3_rel']['min'][i]) / (minmax_values['thumb3_rel']['max'][i] - minmax_values['thumb3_rel']['min'][i]) for i in range(3)])

    df_['thumb4_rel'] = df_['thumb4_rel'].apply(
        lambda x: [(x[i] - minmax_values['thumb4_rel']['min'][i]) / (minmax_values['thumb4_rel']['max'][i] - minmax_values['thumb4_rel']['min'][i]) for i in range(3)])

    df_['press'] = df_['keys'].apply(lambda x: 1 if x.any() else 0)

    df_['coords'] = df_[['thumb3_rel', 'thumb4_rel']].apply(lambda row: list(row['thumb3_rel']) + list(row['thumb4_rel']), axis=1)

    df_ = df_[['coords', 'thumb3_rel', 'thumb4_rel', 'press', 'keys']]

    if to_filter:
        df_ = create_filtered_df(df_)

    return df_


def predict_number_of_typing_press(keyboard_model_: KeyboardModel, df_: pd.DataFrame, device_, prefix_percentage=0.5):
    total_number_of_presses_pred = 0
    total_number_of_presses_gt = 0

    seq_len = keyboard_model_.typing_seq_len

    curr_position = int(len(df_) * prefix_percentage)
    end_position = len(df_)

    while curr_position < end_position - seq_len:
        # Gather the input data
        input_data = df_.iloc[curr_position - seq_len:curr_position]['coords'].tolist()
        input_data = torch.tensor(input_data)
        input_data = input_data.type(torch.float32).to(device_)

        # Predict the next key
        outputs = keyboard_model_.typing_model(input_data)
        next_probability = outputs.squeeze(-1)[-1].detach().cpu()
        next_prediction = (torch.sigmoid(next_probability) > 0.5).item()

        if next_prediction:
            total_number_of_presses_pred += 1

        total_number_of_presses_gt += df_.iloc[curr_position]['press']

        curr_position += 1

    return {'pred': total_number_of_presses_pred, 'gt': total_number_of_presses_gt}


def predict_number_of_new_chars(keyboard_model_: KeyboardModel, df_: pd.DataFrame, device_, prefix_percentage=0.5):
    results_typing_press = predict_number_of_typing_press(keyboard_model_, df_, device_, prefix_percentage)
    typing_press_pred = results_typing_press['pred']
    typing_press_gt = results_typing_press['gt']

    # calculate ratio of typing presses / characters
    ratio = typing_press_gt / len(get_restored_sentence(df_))

    return {'typing_press_pd': typing_press_pred, 'typing_press_gt': typing_press_gt, 'ratio': ratio,
            'new_chars_pred': int(typing_press_pred / ratio), 'new_chars_gt': int(typing_press_gt / ratio)}


def get_tokenization_dict():
    char_to_idx = {'pad':0, 'eos': 1}
    for i in range(0, 255):
        char_to_idx[chr(i)] = i + 2

    idx_to_char = {v: k for k, v in char_to_idx.items()}

    return char_to_idx, idx_to_char


def tokenize_sentence(sentence, char_to_idx):
    return torch.tensor([char_to_idx[char] for char in sentence])


def restore_sentence(keyboard_model_: KeyboardModel, df_: pd.DataFrame, device_, to_concat_gt = False, prefix_percentage=0.5):
    sentence_prefix = get_restored_sentence(df_.iloc[:int(len(df_) * prefix_percentage)])
    restored_sentence = sentence_prefix

    sentence_all = get_restored_sentence(df_)
    curr_sentence_idx = len(sentence_prefix)

    char_to_idx, idx_to_char = get_tokenization_dict()
    sentence_prefix_tokenized = tokenize_sentence(sentence_prefix, char_to_idx)

    input_ids = sentence_prefix_tokenized.view(1, -1).to(device_)  # of shape [batch_size = 1, seq_len]
    results_num_new_chars = predict_number_of_new_chars(keyboard_model_, df_, device_, prefix_percentage)

    if len(sentence_prefix) + results_num_new_chars['new_chars_pred'] > 512:
        new_chars_num = 512 - len(sentence_prefix)
    else:
        new_chars_num = results_num_new_chars['new_chars_pred']

    for i in range(new_chars_num):
        outputs = keyboard_model_.char_model(input_ids)['logits']

        _, predicted = torch.max(outputs, -1)
        predicted = predicted.view(-1)[-1]

        restored_sentence += idx_to_char[predicted.item()]

        if to_concat_gt and curr_sentence_idx < len(sentence_all):
            next_char = char_to_idx[sentence_all[curr_sentence_idx]]
            next_char = torch.tensor([next_char]).view(-1, 1).to(device_)

        else:
            next_char = predicted.view(-1, 1).to(device_)

        input_ids = torch.cat((input_ids, next_char), dim=1)

        curr_sentence_idx += 1

    return {
        'sentence_prefix': sentence_prefix,
        'restored_sentence': restored_sentence,
        'sentence_all': sentence_all,
    }


def calculate_recording_files(data_dir):
    recordings = {}
    for folder_name in os.listdir(data_dir):
        if folder_name.startswith('r'):
            for file_name in os.listdir(f'{data_dir}/{folder_name}'):
                if file_name.endswith('.h5'):
                    if folder_name not in recordings:
                        recordings[folder_name] = []
                    recordings[folder_name].append(f'{data_dir}/{folder_name}/{file_name}')

    return recordings


def evaluate_keyboard_model(args_dict=None):
    args = parse_args(argparse.ArgumentParser())

    if args_dict is not None:
        for key, value in args_dict.items():
            setattr(args, key, value)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keyboard_model_v2 = KeyboardModel(typing_models_version=2, typing_model_criteria=args.model_criteria, device=device)
    keyboard_model_v3 = KeyboardModel(typing_models_version=3, typing_model_criteria=args.model_criteria, device=device)

    recording_files = calculate_recording_files(args.data_dir)
    f = open(args.output_file, 'w')

    def print_and_write(text):
        print(text)
        f.write(text)

    for person in recording_files:
        for recording_file in recording_files[person]:
            df: pd.DataFrame = pd.read_hdf(recording_file)
            df = df[['thumb3', 'thumb4', 'dev_pos', 'keys']]

            sentence = get_restored_sentence(df)
            print_and_write(f'Recording file: {recording_file}\n')
            print_and_write(f'Restored sentence: {sentence}\n')
            print_and_write(f'Prefix sentence: {get_restored_sentence(df.iloc[:int(len(df) * 0.5)])}')
            print_and_write('\n')

            # not filtered
            print_and_write('Not filtered\n')
            print_and_write('-' * 50)

            df_norm = normalize_and_filter_df(df, to_filter=False)
            results_num_new_char = predict_number_of_new_chars(keyboard_model_v3, df_norm, device)

            print_and_write(f'Number of typing presses predicted: {results_num_new_char["typing_press_pd"]}\n')
            print_and_write(f'Number of typing presses ground truth: {results_num_new_char["typing_press_gt"]}\n')
            print_and_write(f'Number of new characters predicted: {results_num_new_char["new_chars_pred"]}\n')
            print_and_write(f'Number of new characters ground truth: {results_num_new_char["new_chars_gt"]}\n')

            print_and_write('To concat with GT\n')
            restored_sentence = restore_sentence(keyboard_model_v3, df_norm, device, to_concat_gt=True)
            print_and_write(f'Restored sentence: {restored_sentence["restored_sentence"]}\n')

            print_and_write('To concat with Predicted\n')
            restored_sentence = restore_sentence(keyboard_model_v3, df_norm, device, to_concat_gt=False)
            print_and_write(f'Restored sentence: {restored_sentence["restored_sentence"]}\n')

            # filtered
            print_and_write('\n')
            print_and_write('Filtered\n')
            print_and_write('-' * 50)

            df_norm = normalize_and_filter_df(df, to_filter=True)
            results_num_new_char = predict_number_of_new_chars(keyboard_model_v2, df_norm, device)

            print_and_write(f'Number of typing presses predicted: {results_num_new_char["typing_press_pd"]}\n')
            print_and_write(f'Number of typing presses ground truth: {results_num_new_char["typing_press_gt"]}\n')
            print_and_write(f'Number of new characters predicted: {results_num_new_char["new_chars_pred"]}\n')
            print_and_write(f'Number of new characters ground truth: {results_num_new_char["new_chars_gt"]}\n')

            print_and_write('To concat with GT\n')
            restored_sentence = restore_sentence(keyboard_model_v2, df_norm, device, to_concat_gt=True)
            print_and_write(f'Restored sentence: {restored_sentence["restored_sentence"]}\n')

            print_and_write('To concat with Predicted\n')
            restored_sentence = restore_sentence(keyboard_model_v2, df_norm, device, to_concat_gt=False)
            print_and_write(f'Restored sentence: {restored_sentence["restored_sentence"]}\n')

            print_and_write('\n')
            print_and_write('\n')

    # save the file
    f.close()


if __name__ == '__main__':
    evaluate_keyboard_model()
