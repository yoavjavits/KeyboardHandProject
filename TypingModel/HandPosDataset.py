from torch.utils.data import Dataset
import os
import pickle
import torch
import pandas as pd
import numpy as np


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


def calculate_minmax_values(recordings, dataset_split):
    """
    Given recordings, a dict with person names as keys and a list of file paths as values, calculate the min and max per person
    :param recordings:
    :return:
    """

    file_path = './data/minmax_values.pkl'

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    minmax_values = {}

    for person in recordings.keys():
        minmax_values[person] = {'thumb3_rel': {'min': [np.inf, np.inf, np.inf], 'max': [-np.inf, -np.inf, -np.inf]},
                                 'thumb4_rel': {'min': [np.inf, np.inf, np.inf], 'max': [-np.inf, -np.inf, -np.inf]}}

        for recording in recordings[person]:
            df: pd.DataFrame = pd.read_hdf(path_or_buf=recording, key='df')

            # Split the data
            if dataset_split == 'train':
                start = 0
                end = int(0.8 * len(df))
            elif dataset_split == 'valid':
                start = int(0.8 * len(df))
                end = int(0.9 * len(df))
            else:
                start = int(0.9 * len(df))
                end = len(df)

            df = df.iloc[start:end]

            # Get the relative positions
            df['thumb3_rel'] = df['thumb3'] - df['dev_pos']
            df['thumb4_rel'] = df['thumb4'] - df['dev_pos']

            # Create separation between x, y and z values
            for pos in range(3):
                df[f'thumb3_rel_{pos}'] = df['thumb3_rel'].apply(lambda x: x[pos])
                df[f'thumb4_rel_{pos}'] = df['thumb4_rel'].apply(lambda x: x[pos])

            # Calculate the min and max values
            for pos in range(3):
                minmax_values[person]['thumb3_rel']['min'][pos] = min(minmax_values[person]['thumb3_rel']['min'][pos], min(df[f'thumb3_rel_{pos}']))
                minmax_values[person]['thumb3_rel']['max'][pos] = max(minmax_values[person]['thumb3_rel']['max'][pos], max(df[f'thumb3_rel_{pos}']))

                minmax_values[person]['thumb4_rel']['min'][pos] = min(minmax_values[person]['thumb4_rel']['min'][pos], min(df[f'thumb4_rel_{pos}']))
                minmax_values[person]['thumb4_rel']['max'][pos] = max(minmax_values[person]['thumb4_rel']['max'][pos], max(df[f'thumb4_rel_{pos}']))

    return minmax_values


def create_dfs_list(dataset_split, data_dir, filtered=False):
    dfs_list = []

    recordings = calculate_recording_files(data_dir)
    minmax_values = calculate_minmax_values(recordings, dataset_split)

    print('Start calculating the dataframes', flush=True)
    curr_df = 0
    total_dfs = sum([len(recordings[person]) for person in recordings.keys()])

    for person in recordings.keys():
        minmax_values_person = minmax_values[person]

        for recording in recordings[person]:
            df: pd.DataFrame = pd.read_hdf(path_or_buf=recording, key='df')

            # Get the relative positions
            df['thumb3_rel'] = df['thumb3'] - df['dev_pos']
            df['thumb4_rel'] = df['thumb4'] - df['dev_pos']

            # Normalize the relative positions
            df['thumb3_rel'] = df['thumb3_rel'].apply(
                lambda x: [(x[i] - minmax_values_person['thumb3_rel']['min'][i]) / (minmax_values_person['thumb3_rel']['max'][i] - minmax_values_person['thumb3_rel']['min'][i]) for i in range(3)])

            df['thumb4_rel'] = df['thumb4_rel'].apply(
                lambda x: [(x[i] - minmax_values_person['thumb4_rel']['min'][i]) / (minmax_values_person['thumb4_rel']['max'][i] - minmax_values_person['thumb4_rel']['min'][i]) for i in range(3)])

            df['press'] = df['keys'].apply(lambda x: 1 if x.any() else 0)

            df['coords'] = df[['thumb3_rel', 'thumb4_rel']].apply(lambda row: list(row['thumb3_rel']) + list(row['thumb4_rel']), axis=1)

            # Split the data
            if dataset_split == 'train':
                start = 0
                end = int(0.8 * len(df))
            elif dataset_split == 'valid':
                start = int(0.8 * len(df))
                end = int(0.9 * len(df))
            else:
                start = int(0.9 * len(df))
                end = len(df)

            df = df.iloc[start:end].reset_index(drop=True)
            df = df[['coords', 'thumb3_rel', 'thumb4_rel', 'press']]

            # Filter the data if needed
            df = create_filtered_df(df) if filtered else df

            # Append data
            dfs_list.append(df)

            # update the progress
            curr_df += 1
            if int(curr_df / total_dfs * 100) % 10 == 0:
                print(f'Loading data: {int(curr_df / total_dfs * 100)}%', flush=True)

    return dfs_list


def calculate_total_length(dfs_list, sequence_length):
    total_length = 0

    for df in dfs_list:
        total_length += len(df) - sequence_length

    return total_length


class HandPosDataset(Dataset):
    def __init__(self, dataset_split, dataset_type, sequence_length=10, data_dir='../../Data', filtered=False):
        if dataset_split not in ['train', 'valid', 'test']:
            raise ValueError('dataset_type must be either train or test')

        if dataset_type not in ['many', 'one']:
            raise ValueError('dataset_type must be either many or one')

        self.sequence_length = sequence_length
        self.df_list = create_dfs_list(dataset_split, data_dir, filtered)
        self.length = calculate_total_length(self.df_list, self.sequence_length)
        self.dataset_type = dataset_type

    def update_seq_length(self, sequence_length):
        self.sequence_length = sequence_length
        self.length = calculate_total_length(self.df_list, self.sequence_length)

    def update_dataset_type(self, dataset_type):
        if dataset_type not in ['many', 'one']:
            raise ValueError('dataset_type must be either many or one')

        self.dataset_type = dataset_type

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # find the correct df
        df_index = 0
        for df in self.df_list:
            if idx < len(df) - self.sequence_length:
                break
            idx -= len(df) - self.sequence_length
            df_index += 1

        # get the data
        df = self.df_list[df_index]
        input_data = df.iloc[idx:idx+self.sequence_length]['coords'].tolist()
        input_data = torch.tensor(input_data)

        if self.dataset_type == 'many':
            target_data = df.iloc[idx:idx+self.sequence_length]['press'].to_numpy()
            target_data = torch.tensor(target_data)

        elif self.dataset_type == 'one':
            target_data = 1 if df.iloc[idx+self.sequence_length]['press'] == 1 else 0
            target_data = torch.tensor(target_data)

        else:
            raise ValueError('dataset_type must be either many or one')

        return input_data, target_data


if __name__ == '__main__':
    print('Calculate min-max values', flush=True)

    # recordings = calculate_recording_files('../../Data')
    # minmax_values = calculate_minmax_values(recordings, 'train')
    #
    # with open('./data/minmax_values.pkl', 'wb') as f:
    #     pickle.dump(minmax_values, f)

    print('Create the datasets', flush=True)

    print('Train dataset', flush=True)
    train_dataset = HandPosDataset('train', 'many', filtered=True)
    print('Valid dataset', flush=True)
    valid_dataset = HandPosDataset('valid', 'many', filtered=True)
    print('Test dataset', flush=True)
    test_dataset = HandPosDataset('test', 'many', filtered=True)

    with open('./data/filtered/train.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open('./data/filtered/valid.pkl', 'wb') as f:
        pickle.dump(valid_dataset, f)

    with open('./data/filtered/test.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)

    print('Datasets created and saved', flush=True)
