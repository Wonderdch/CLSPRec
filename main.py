import datetime
import random
import numpy as np

import torch
import pickle
import time
import os
import pandas as pd

import settings
from CLSPRec import CLSPRec
from results.data_reader import print_output_to_file, calculate_average, clear_log_meta_model

device = settings.gpuId if torch.cuda.is_available() else 'cpu'
city = settings.city

if settings.enable_ssl and settings.enable_distance_sample:
    df_farthest_POIs = pd.read_csv(f"./raw_data/{city}_farthest_POIs.csv")


def generate_sample_to_device(sample):
    sample_to_device = []
    if settings.enable_dynamic_day_length:
        last_day = sample[-1][5][0]
        for seq in sample:
            seq_day = seq[5][0]
            if last_day - seq_day < settings.sample_day_length:
                features = torch.tensor(seq[:5]).to(device)
                day_nums = torch.tensor(seq[5]).to(device)
                sample_to_device.append((features, day_nums))
    else:
        for seq in sample:
            features = torch.tensor(seq[:5]).to(device)
            day_nums = torch.tensor(seq[5]).to(device)
            sample_to_device.append((features, day_nums))

    return sample_to_device


def generate_day_sample_to_device(day_trajectory):
    features = torch.tensor(day_trajectory[:5]).to(device)
    day_nums = torch.tensor(day_trajectory[5]).to(device)
    day_to_device = (features, day_nums)
    return day_to_device


def generate_negative_sample_list(dataset, user_id, current_POI):
    k = settings.neg_sample_count
    neg_day_sample_to_device_list = []
    if settings.enable_distance_sample:
        # Random sample k negative samples from other users' trajectories
        # and the negative samples contain the farthest POIs from current POI
        farthest_POIs = set(df_farthest_POIs.iloc[current_POI].values.tolist())
        eligible_sequences = []
        for seq in dataset:
            if seq[0][2][0] != user_id:
                for i in range(len(seq)):
                    if set(seq[i][0]).intersection(farthest_POIs):
                        eligible_sequences.append(seq[i])

        if len(eligible_sequences) == 0:
            print(f'Can not find eligible_sequences, current POI {current_POI}')
        elif len(eligible_sequences) < k:
            neg_day_samples = eligible_sequences
        else:
            neg_day_samples = random.sample(eligible_sequences, k)
    else:
        # Random sample k negative samples from other users' trajectories
        neg_day_samples = random.sample([seq[-1] for seq in dataset if seq[0][2][0] != user_id], k)

    for neg_day_sample in neg_day_samples:
        neg_day_sample_to_device_list.append(generate_day_sample_to_device(neg_day_sample))
    return neg_day_sample_to_device_list


def train_model(train_set, test_set, h_params, vocab_size, device, run_name):
    torch.cuda.empty_cache()
    model_path = f"./results/{run_name}_model"
    log_path = f"./results/{run_name}_log"
    meta_path = f"./results/{run_name}_meta"

    print("parameters:", h_params)

    if os.path.isfile(f'./results/{run_name}_model'):
        try:
            os.remove(f"./results/{run_name}_meta")
            os.remove(f"./results/{run_name}_model")
            os.remove(f"./results/{run_name}_log")
        except OSError:
            pass
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()

    # construct model
    rec_model = CLSPRec(
        vocab_size=vocab_size,
        f_embed_size=h_params['embed_size'],
        num_encoder_layers=h_params['tfp_layer_num'],
        num_lstm_layers=h_params['lstm_layer_num'],
        num_heads=h_params['head_num'],
        forward_expansion=h_params['expansion'],
        dropout_p=h_params['dropout']
    )

    rec_model = rec_model.to(device)

    # Continue with previous training
    start_epoch = 0
    if os.path.isfile(model_path):
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()

        meta_file = open(meta_path, "rb")
        start_epoch = pickle.load(meta_file) + 1
        meta_file.close()

    params = list(rec_model.parameters())

    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}

    for epoch in range(start_epoch, h_params['epoch']):
        begin_time = time.time()
        total_loss = 0.
        for sample in train_set:
            sample_to_device = generate_sample_to_device(sample)

            neg_sample_to_device_list = []
            if settings.enable_ssl:
                user_id = sample[0][2][0]
                current_POI = sample[-1][0][-2]
                neg_sample_to_device_list = generate_negative_sample_list(train_set, user_id, current_POI)

            loss, _ = rec_model(sample_to_device, neg_sample_to_device_list)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Test
        recall, ndcg, map = test_model(test_set, rec_model)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map

        # Record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[epoch] = avg_loss
        print(f"epoch: {epoch}; average loss: {avg_loss}, time taken: {int(time.time() - begin_time)}s")
        # Save model
        torch.save(rec_model.state_dict(), model_path)
        # Save last epoch
        meta_file = open(meta_path, 'wb')
        pickle.dump(epoch, meta_file)
        meta_file.close()

        # Early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss) > 10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {epoch}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        file.close()

    print("============================")


def test_model(test_set, rec_model, ks=[1, 5, 10]):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    for sample in test_set:
        sample_to_device = generate_sample_to_device(sample)

        neg_sample_to_device_list = []
        if settings.enable_ssl:
            user_id = sample[0][2][0]
            current_POI = sample[-1][0][-2]
            neg_sample_to_device_list = generate_negative_sample_list(test_set, user_id, current_POI)

        pred, label = rec_model.predict(sample_to_device, neg_sample_to_device_list)
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")

    return recalls, NDCGs, MAPs


if __name__ == '__main__':
    # Get current time
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Datetime of now：", now_str)

    # Get parameters
    h_params = {
        'expansion': 4,
        'random_mask': settings.enable_random_mask,
        'mask_prop': 0.1,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'loss_delta': 1e-3}

    processed_data_directory = './processed_data/'
    if settings.enable_dynamic_day_length:
        processed_data_directory += 'dynamic_day_length'
    else:
        processed_data_directory += 'original'

    # Read training data
    file = open(f"{processed_data_directory}/{city}_train", 'rb')
    train_set = pickle.load(file)
    file = open(f"{processed_data_directory}/{city}_valid", 'rb')
    valid_set = pickle.load(file)

    # Read meta data
    file = open(f"{processed_data_directory}/{city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {"POI": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device)}

    # Adjust specific parameters for each city
    if city == 'SIN':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 3
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
    elif city == 'NYC':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 1
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.1
        h_params['head_num'] = 1
    elif city == 'PHO':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 4
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1

    # Create output folder
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    print(f'Current GPU {settings.gpuId}')
    for run_num in range(1, 1 + settings.run_times):
        run_name = f'{settings.output_file_name} {run_num}'
        print(run_name)

        train_model(train_set, valid_set, h_params, vocab_size, device, run_name=run_name)
        print_output_to_file(settings.output_file_name, run_num)

        t = random.randint(1, 9)
        print(f"sleep {t} seconds")
        time.sleep(t)

        clear_log_meta_model(settings.output_file_name, run_num)
    calculate_average(settings.output_file_name, settings.run_times)
