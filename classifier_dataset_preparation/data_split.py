import os
import sys
import pandas as pd


def data_split(data_dic_path, split):
    # splits data into training and testing

    data_folder = os.path.dirname(os.path.realpath(data_dic_path))
    df = pd.read_csv(data_dic_path, sep=',', names=['class_id', 'dir'])
    print('Original df: ', len(df))

    samples_per_class_df = df.groupby('class_id', as_index=True).count()

    df_list_train = []
    df_list_test = []
    for class_id, total_samples_class in enumerate(
            samples_per_class_df['dir']):
        train_samples_class = int(total_samples_class * split[0])
        test_samples_class = total_samples_class - train_samples_class
        assert (train_samples_class +
                test_samples_class == total_samples_class)
        train_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').head(train_samples_class)
        test_subset_class = df.loc[df['class_id'] == class_id].groupby(
            'class_id').tail(test_samples_class)
        df_list_train.append(train_subset_class)
        df_list_test.append(test_subset_class)

    df_train = pd.concat(df_list_train)
    df_test = pd.concat(df_list_test)

    print('Train df: ')
    print(df_train.head())
    print(df_train.shape)
    print('Test df: ')
    print(df_test.head())
    print(df_test.shape)

    df_train_name = os.path.join(data_folder, 'train.csv')
    df_train.to_csv(df_train_name, sep=',', header=False, index=False)

    df_test_name = os.path.join(data_folder, 'test.csv')
    df_test.to_csv(df_test_name, sep=',', header=False, index=False)
    print('Finished saving train and test split dictionaries.')


def main():

    data_dic_path = os.path.abspath(sys.argv[1])
    split_train = float(sys.argv[2])  # % of data for train (def: 0.8)
    split_test = float(sys.argv[3])  # % of data for val (def: 0.2)
    assert split_train + split_test == 1, 'Arguments for split ratios should add up to 1'
    split = [split_train, split_test]

    print(split)
    data_split(data_dic_path, split)


main()
