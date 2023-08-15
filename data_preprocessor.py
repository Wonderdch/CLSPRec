import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta
import random
import settings

# Constants
min_seq_len = 3
min_seq_num = 3
min_short_term_len = 5
min_long_term_count = 2
pre_seq_window = 7
random_seed = 2021


# Split sequence
def generate_sequence(input_data, min_seq_len, min_seq_num):
    """Split and filter action sequences for each user

    Args:
        input_data (DataFrame): raw data read from csv file
        min_seq_len (int): minimum length for a sequence to be considered valid
        min_seq_num (int): minimum no. sequences for a user to be considered valid

    Returns:
        total_sequences_dict ({user_id: [[visit_id]]}): daily action sequences for each user
        total_sequences_meta ([(user)[(timestamp(int),seq_len(int))]]) : date and length for each sequnece
    """

    def _remove_consecutive_visit(visit_record):
        """remove duplicated consecutive visits in a sequence

        Args:
            visit_record (DataFrame): raw sequences

        Returns:
            clean_sequence (list): sequences with no duplicated consecutive visits
        """
        clean_sequence = []
        for index, _ in visit_record.iterrows():
            clean_sequence.append(index)
        return clean_sequence

    total_sequences_dict = {}  # records visit id in each sequence
    total_sequences_meta = []  # records sequence date and length

    seq_count = 0  # for statistics only

    input_data['Local_sg_time'] = pd.to_datetime(input_data['Local_Time_True'])

    for user in input_data['UserId'].unique():  # process sequences for each user
        user_visits = input_data[input_data['UserId'] == user]
        user_sequences, user_sequences_meta = [], []
        unique_date_group = user_visits.groupby([user_visits['Local_sg_time'].dt.date])
        for date in unique_date_group.groups:  # process sequences on each day
            single_date_visit = unique_date_group.get_group(date)
            single_sequence = _remove_consecutive_visit(single_date_visit)
            if len(single_sequence) >= min_seq_len:  # filter sequences too short
                user_sequences.append(single_sequence)
                user_sequences_meta.append((date, len(single_sequence)))
                seq_count += 1
        if len(user_sequences) >= min_seq_num:  # filter users with too few visits
            total_sequences_dict[user] = np.array(user_sequences, dtype=object)
            total_sequences_meta.append(user_sequences_meta)
    print(f"Generated {seq_count} sequences in total for {len(total_sequences_dict.keys())} users")
    return total_sequences_dict, total_sequences_meta


# Generate sequences of different features
def _reIndex_3d_list(input_list):
    """Reindex the elements in sequences

    Args:
        input_list (nd_array: [(all users)[(user list)[(seq list)]]]): a 3d list containing all sequences for all users

    Returns:
        reIndexed_list (3d list): reindexed list
        index_map ([id]): each element is an original id, the index of an element is the new index the id
    """

    def _flatten_3d_list(input_list):
        """flattern a 3d list to 1d

        Args:
            input_list (nd_array: [(all users)[(user list)[(seq list)]]]): a 3d list containing all sequences for all users

        Returns:
            1d-list: flattened list
        """
        twoD_lists = input_list.flatten()
        return np.hstack([np.hstack(twoD_list) for twoD_list in twoD_lists])

    def _old_id_to_new(mapping, old_id):
        """convert old_id to new index by mapping given

        Args:
            mapping ([id]): each element is an original id, the index of an element is the new index the id
            old_id (Any): the original token/id in the list

        Returns:
            int: new index of the token
        """
        return np.where(mapping == old_id)[0].flat[0]

    flat_list = _flatten_3d_list(input_list)  # make 3d list 1d
    index_map = np.unique(flat_list)  # get
    reIndexed_list = []
    for user_seq in input_list:  # seq list for each user
        reIndexed_user_list = []
        for seq in user_seq:  # each seq
            reIndexed_user_list.append([_old_id_to_new(index_map, poi) for poi in seq])
        reIndexed_list.append(reIndexed_user_list)
    reIndexed_list = np.array(reIndexed_list, dtype=object)

    return reIndexed_list, index_map


def generate_POI_sequences(input_data, visit_sequence_dict):
    """generate location transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_POI_sequences (nd_array: [[[POI_index]]]): daily location transition sequences for each user
        POI_reIndex_mapping ([POI_id]): index is the new POI index and element is the original POI id
    """
    POI_sequences = []

    for user in visit_sequence_dict:
        user_POI_sequences = []
        for seq in visit_sequence_dict[user]:
            single_POI_sequence = []
            for visit in seq:
                single_POI_sequence.append(input_data['VenueId'][visit])
            user_POI_sequences.append(single_POI_sequence)
        POI_sequences.append(user_POI_sequences)
    reIndexed_POI_sequences, POI_reIndex_mapping = _reIndex_3d_list(np.array(POI_sequences, dtype=object))
    return reIndexed_POI_sequences, POI_reIndex_mapping


def generate_category_sequences(input_data, visit_sequence_dict):
    """generate category transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_cat_sequences (nd_array: [[[cat_index]]]): daily category transition sequences for each user
        cat_reIndex_mapping ([cat_name]): index is the new category index and element is the original category name
    """
    cat_sequences = []
    for user in visit_sequence_dict:
        user_cat_sequences = []
        for seq in visit_sequence_dict[user]:
            single_cat_sequence = []
            for visit in seq:
                single_cat_sequence.append(input_data['L1_Category'][visit])
            user_cat_sequences.append(single_cat_sequence)
        cat_sequences.append(user_cat_sequences)
    reIndexed_cat_sequences, cat_reIndex_mapping = _reIndex_3d_list(np.array(cat_sequences, dtype=object))
    return reIndexed_cat_sequences, cat_reIndex_mapping


def generate_user_sequences(input_data, visit_sequence_dict):
    """generate time (in hour) transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_user_sequences (nd_array: [[[user_index]]]): daily user sequences (same for each sequence)
        user_reIndex_mapping ([user_id]): index is the new user index and element is the original user id
    """
    all_user_sequences = []
    for user in visit_sequence_dict:
        user_sequences = []
        for seq in visit_sequence_dict[user]:
            single_user_sequence = [user] * len(seq)
            user_sequences.append(single_user_sequence)
        all_user_sequences.append(user_sequences)
    reIndexed_user_sequences, user_reIndex_mapping = _reIndex_3d_list(np.array(all_user_sequences, dtype=object))
    return reIndexed_user_sequences, user_reIndex_mapping


def generate_hour_sequences(input_data, visit_sequence_dict):
    """generate time (in hour) transition sequences

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_hour_sequences (nd_array: [[[time_index]]]): daily hour transition sequences for each user
        hour_reIndex_mapping ([hour]): index is the new hour index and element is the original hour
    """
    input_data["hour"] = pd.to_datetime(input_data['Local_Time_True']).dt.hour  # add hour column in raw data

    hour_sequences = []
    for user in visit_sequence_dict:
        user_hour_sequences = []
        for seq in visit_sequence_dict[user]:
            single_hour_sequence = []
            for visit in seq:
                single_hour_sequence.append(input_data['hour'][visit])
            user_hour_sequences.append(single_hour_sequence)
        hour_sequences.append(user_hour_sequences)
    reIndexed_hour_sequences, hour_reIndex_mapping = _reIndex_3d_list(np.array(hour_sequences, dtype=object))
    return reIndexed_hour_sequences, hour_reIndex_mapping


def generate_day_sequences(input_data, visit_sequence_dict):
    """generate weekday/weekend tag for each sequence

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_day_sequences (nd_array: [[[day_index]]]): daily weekday/weekend sequences (same for each sequence)
        day_reIndex_mapping ([weekday/weekend]): index is the new day index and element is the original weekday(False)/weekend(True) tag
    """
    input_data["is_weekend"] = pd.to_datetime(
        input_data['Local_Time_True']).dt.dayofweek > 4  # add hour column in raw data

    day_sequences = []
    for user in visit_sequence_dict:
        user_day_sequences = []
        for seq in visit_sequence_dict[user]:
            single_day_sequence = []
            for visit in seq:
                single_day_sequence.append(input_data['is_weekend'][visit])
            user_day_sequences.append(single_day_sequence)
        day_sequences.append(user_day_sequences)
    reIndexed_day_sequences, day_reIndex_mapping = _reIndex_3d_list(np.array(day_sequences, dtype=object))
    return reIndexed_day_sequences, day_reIndex_mapping


def generate_date_sequences(input_data, visit_sequence_dict):
    """generate date for each sequence

    Args:
        input_data (DataFrame): raw check-in data
        visit_sequence_dict ({user_id: [[visit_id]]}): daily action sequences for each user

    Returns:
        reIndexed_date_sequences (nd_array: [[[day_index]]]): date sequences (same for each sequence)
    """
    input_data["Local_Date"] = pd.to_datetime(
        input_data['Local_Time_True']).dt.date

    date_sequences = []
    for user in visit_sequence_dict:
        user_date_sequences = []
        for seq in visit_sequence_dict[user]:
            local_date = input_data['Local_Date'][seq[0]]
            single_day_sequence = [local_date] * len(seq)
            user_date_sequences.append(single_day_sequence)
        date_sequences.append(user_date_sequences)
    reIndexed_date_sequences, _ = _reIndex_3d_list(np.array(date_sequences, dtype=object))
    return reIndexed_date_sequences


# Generate (short term + long term) feed data
def filter_long_short_term_sequences(total_sequences_meta, min_short_term_len, pre_seq_window, min_long_term_count):
    """filter valid long+short-term sequences for generation of input data
        criteria: 1. the feed data is composed of multiple long-term sequences and one short-term sequence;
                  2. the short term sequence length >= min_short_term_len(5)
                  3. the long term sequences a sequences 7 days before the short term sequence
                  4. the number of long term sequences must >= min_long_term_count(2)

    Args:
        total_sequences_meta ([(user)[(timestamp(int),seq_len(int))]]): date and length for each sequnece
        min_short_term_len (int): minimum visits in a short-term sequence
        pre_seq_window (int): number of days to look for long-term sequences
        min_long_term_count (int): minimum number of long-term sequences to make the long-short sequences valid

    Returns:
        valid_input_index ([(all users)[(each user)[(valid sequences)seq_index]]]): valid long+short term sequneces for each user
    """

    valid_input_index = []  # filtered long+short term data
    valid_user_count, valid_input_count = 0, 0,  # for statistics purpose

    for _, user_sequences in enumerate(total_sequences_meta):  # for each user
        user_valid_input_index = []
        for seq_index, seq in enumerate(user_sequences):  # for each sequence
            seq_time, seq_len = seq[0], seq[1]
            if seq_len >= min_short_term_len:  # valid short-term sequence
                start_time, end_time = seq_time - timedelta(days=pre_seq_window), seq_time
                long_term_seqs = [(index, seq) for index, seq in enumerate(user_sequences[:seq_index]) if
                                  start_time <= seq[0] <= end_time]
                if len(long_term_seqs) >= min_long_term_count:  # valid long-short term sequence
                    user_valid_input_index.append([seq[0] for seq in long_term_seqs] + [seq_index])
                    valid_input_count += 1
        valid_input_index.append(user_valid_input_index)
        valid_user_count += 1 if len(user_valid_input_index) > 0 else 0

    print(f"Filtered {valid_input_count} valid input long+short term sequences for {valid_user_count} users.")
    return valid_input_index


def filter_long_short_term_sequences_for_dynamic_day_length(total_sequences_meta, min_short_term_len, pre_seq_window,
                                                            min_long_term_count):
    valid_input_index = []  # filtered long+short term data
    valid_user_count, valid_input_count = 0, 0,  # for statistics purpose

    for _, user_sequences in enumerate(total_sequences_meta):  # for each user
        user_valid_input_index = []
        for seq_index, seq in enumerate(user_sequences):  # for each sequence
            seq_time, seq_len = seq[0], seq[1]
            if seq_len >= min_short_term_len:  # valid short-term sequence
                start_time, end_time = seq_time - timedelta(days=pre_seq_window), seq_time

                three_days_seqs = [(index, seq) for index, seq in enumerate(user_sequences[:seq_index]) if
                                   (seq_time - timedelta(days=3)) <= seq[0] <= end_time]

                if len(three_days_seqs) >= min_long_term_count:  # valid long-short term sequence
                    long_term_seqs = [(index, seq) for index, seq in enumerate(user_sequences[:seq_index]) if
                                      start_time <= seq[0] <= end_time]
                    user_valid_input_index.append([seq[0] for seq in long_term_seqs] + [seq_index])
                    valid_input_count += 1
        valid_input_index.append(user_valid_input_index)
        valid_user_count += 1 if len(user_valid_input_index) > 0 else 0

    print(f"Filtered {valid_input_count} valid input long+short term sequences for {valid_user_count} users.")
    return valid_input_index


def generate_input_samples(feature_sequences, valid_input_index):
    """turn a feature sequence into a input long+short term data to be fed into model

    Args:
        feature_sequences (nd_array: [[[id]]]): daily transition sequences for each user for certain feature
        valid_input_index ([(all users)[(each user)[(valid sequences)seq_index]]]): valid long+short term sequneces for each user

    Return:
        input_samples ([(input sample)[(sequences)feature_id]]]): valid long+short term feature sequences for each user
    """
    input_samples = []
    for user_index, user_sequences in enumerate(valid_input_index):
        if len(user_sequences) != 0:
            for seq in user_sequences:
                feature_sequence = [feature_sequences[user_index][index] for index in seq]
                input_samples.append(feature_sequence)
    return input_samples


def split_train_test(input_samples):
    """split a input sequence into training, validation and testing sequences
        criteria: train-80%, validation-10%, test-10%

    Args:
        input_samples (3d array: [(each sample)[(valid sequences)feature_id]]): valid long+short term feature sequences for each user

    Returns:
        all_training_samples: 80% of samples for training
        all_validation_samples: 10% of samples for validation
        all_testing_samples: 10% of samples for testing
        all_training_validation_samples: 90% of samples for final training after validation
    """
    random.Random(random_seed).shuffle(input_samples)
    N = len(input_samples)
    train_valid_boundary = int(0.8 * N)
    valid_test_boundary = int(0.9 * N)
    all_training_samples = input_samples[:train_valid_boundary]
    all_validation_samples = input_samples[train_valid_boundary:valid_test_boundary]
    all_testing_samples = input_samples[valid_test_boundary:]
    all_training_validation_samples = input_samples[:valid_test_boundary]

    return all_training_samples, all_validation_samples, all_testing_samples, all_training_validation_samples


def reshape_data(original_data):
    """combine different samples for each features to one sample containing all features

    Args:
        original_data ([features * sample * sequence]): combination of samples for each feature

    Return:
        reshaped_data ([sample * sequence * features]): each sample contains myltiple features
    """
    result = []

    samples = np.transpose(np.array(original_data, dtype=object), (1, 0))

    for sample in samples:
        sample_data = []
        feature_num = len(sample)
        sequence_num = len(sample[0])
        for i in range(sequence_num):
            sample_data.append([sample[j][i] for j in range(feature_num)])
        result.append(sample_data)
    return result


def dump_data(data, city, data_type):
    """save data as pickle file

    Args:
        data ([(feature)[(sample)[feature_id]]]): processed data
        city (str): city code for file naming
        data_type (str): data description for file naming
    """
    directory = './processed_data'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = directory + "/{}_{}"

    pickle.dump(data, open(file_path.format(city, data_type), 'wb'))


# Completely process data for one city
def generate_data(city):
    """
    Generate complete train and test data set for one city
        Save the result in pickle files
    Args:
        city (str): city to read data from and process
    """
    print(f"******Process data for {city}******")

    data = pd.read_csv(f"./raw_data/{city}_checkin_with_active_regionId.csv")

    visit_sequence_dict, total_sequences_meta = generate_sequence(data, min_seq_len, min_seq_num)

    if settings.enable_dynamic_day_length:
        valid_input_index = filter_long_short_term_sequences_for_dynamic_day_length(total_sequences_meta,
                                                                                    min_short_term_len, pre_seq_window,
                                                                                    min_long_term_count)
    else:
        valid_input_index = filter_long_short_term_sequences(total_sequences_meta, min_short_term_len, pre_seq_window,
                                                             min_long_term_count)

    train_data, valid_data, test_data, train_valid_data, meta_data = [], [], [], [], {}

    # POI inputs
    poi_sequences, poi_mapping = generate_POI_sequences(data, visit_sequence_dict)
    poi_input_data = generate_input_samples(poi_sequences, valid_input_index)
    poi_train, poi_valid, poi_test, poi_train_valid = split_train_test(poi_input_data)
    train_data.append(poi_train)
    valid_data.append(poi_valid)
    test_data.append(poi_test)
    train_valid_data.append(poi_train_valid)
    print("poi sequence generated.")

    # Category inputs
    cat_sequences, cat_mapping = generate_category_sequences(data, visit_sequence_dict)
    cat_input_data = generate_input_samples(cat_sequences, valid_input_index)
    cat_train, cat_valid, cat_test, cat_train_valid = split_train_test(cat_input_data)
    train_data.append(cat_train)
    valid_data.append(cat_valid)
    test_data.append(cat_test)
    train_valid_data.append(cat_train_valid)
    print("category sequence generated.")

    # User inputs
    user_sequences, user_mapping = generate_user_sequences(data, visit_sequence_dict)
    user_input_data = generate_input_samples(user_sequences, valid_input_index)
    user_train, user_valid, user_test, user_train_valid = split_train_test(user_input_data)
    train_data.append(user_train)
    valid_data.append(user_valid)
    test_data.append(user_test)
    train_valid_data.append(user_train_valid)
    print("user sequence generated.")

    # Hour inputs
    hour_sequences, hour_mapping = generate_hour_sequences(data, visit_sequence_dict)
    hour_input_data = generate_input_samples(hour_sequences, valid_input_index)
    hour_train, hour_valid, hour_test, hour_train_valid = split_train_test(hour_input_data)
    train_data.append(hour_train)
    valid_data.append(hour_valid)
    test_data.append(hour_test)
    train_valid_data.append(hour_train_valid)
    print("hour sequence generated.")

    # Day inputs
    day_sequences, day_mapping = generate_day_sequences(data, visit_sequence_dict)
    day_input_data = generate_input_samples(day_sequences, valid_input_index)
    day_train, day_valid, day_test, day_train_valid = split_train_test(day_input_data)
    train_data.append(day_train)
    valid_data.append(day_valid)
    test_data.append(day_test)
    train_valid_data.append(day_train_valid)
    print("day sequence generated.")

    # Date inputs
    date_sequences = generate_date_sequences(data, visit_sequence_dict)
    date_input_data = generate_input_samples(date_sequences, valid_input_index)
    date_train, date_valid, date_test, date_train_valid = split_train_test(date_input_data)
    train_data.append(date_train)
    valid_data.append(date_valid)
    test_data.append(date_test)
    train_valid_data.append(date_train_valid)
    print("date sequence generated.")

    # Reshape data: [features * sample * sequence] -> [sample * sequence * features]
    train_data = reshape_data(train_data)
    valid_data = reshape_data(valid_data)
    test_data = reshape_data(test_data)
    train_valid_data = reshape_data(train_valid_data)

    # Meta data
    meta_data["POI"] = poi_mapping
    meta_data["cat"] = cat_mapping
    meta_data["user"] = user_mapping
    meta_data["hour"] = hour_mapping
    meta_data["day"] = day_mapping

    # Output data
    dump_data(train_data, city, "train")
    dump_data(valid_data, city, "valid")
    dump_data(test_data, city, "test")
    dump_data(train_valid_data, city, "train_valid")
    dump_data(meta_data, city, "meta")


if __name__ == '__main__':
    city_list = ['PHO', 'NYC', 'SIN']

    if settings.enable_dynamic_day_length:
        pre_seq_window = 14
    else:
        pre_seq_window = 7

    for city in city_list:
        generate_data(city)
