import json
import sys

from collections import defaultdict
from math import sqrt

import numpy as np
import torch

from .rbm import CFRBM
from .device import parse_args
from .experiments import read_experiment
from .utils import revert_expected_value, iteration_str
from .dataset import load_dataset
from .preparation import PreparedProfiles


def run(name, dataset, config, all_users, all_movies, tests, initial_v, sep, device=None):
    config_name = config['name']
    number_hidden = config['number_hidden']
    epochs = config['epochs']
    ks = config['ks']
    momentums = config['momentums']
    l_w = config['l_w']
    l_v = config['l_v']
    l_h = config['l_h']
    decay = config['decay']
    batch_size = config['batch_size']

    config_result = config.copy()
    config_result['results'] = []

    rbm = CFRBM(len(all_movies) * 5, number_hidden, device=device)
    print(f"Using device: {rbm.device}")

    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            profiles[uid].append((mid, float(rat)))

    prepared = PreparedProfiles(profiles, all_movies, tests)
    del profiles

    print("Users and ratings loaded")

    for j in range(epochs):
        def get_index(col):
            if j // (epochs // len(col)) < len(col):
                return j // (epochs // len(col))
            else:
                return -1

        index = get_index(ks)
        mindex = get_index(momentums)
        icurrent_l_w = get_index(l_w)
        icurrent_l_v = get_index(l_v)
        icurrent_l_h = get_index(l_h)

        k = ks[index]
        momentum = momentums[mindex]
        current_l_w = l_w[icurrent_l_w]
        current_l_v = l_v[icurrent_l_v]
        current_l_h = l_h[icurrent_l_h]

        train = rbm.cdk_fun(None,
                            None,
                            k=k,
                            w_lr=current_l_w,
                            v_lr=current_l_v,
                            h_lr=current_l_h,
                            decay=decay,
                            momentum=momentum)
        predict = rbm.predict()

        for train_batch, train_masks in prepared.training_batches(j, batch_size):
            train(torch.from_numpy(train_batch).to(rbm.device),
                  torch.from_numpy(train_masks).to(rbm.device))
            sys.stdout.write('.')
            sys.stdout.flush()

        ratings = []
        predictions = []

        for batch, test_batch in prepared.test_batches(batch_size):
            positions = {profile_id: pos for pos, profile_id in enumerate(batch)}
            user_preds = revert_expected_value(
                predict(torch.from_numpy(test_batch).to(rbm.device)).cpu().numpy())
            for profile_id in batch:
                test_movies = tests[profile_id]
                try:
                    for movie, rating in test_movies:
                        current_profile = user_preds[positions[profile_id]]
                        predicted = current_profile[prepared.visible_index[movie]]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        vabs = np.vectorize(abs)
        distances = np.array(ratings) - np.array(predictions)

        mae = vabs(distances).mean()
        rmse = sqrt((distances ** 2).mean())

        iteration_result = {
            'iteration': j,
            'k': k,
            'momentum': momentum,
            'mae': mae,
            'rmse': rmse,
            'lrate': current_l_w
        }

        config_result['results'].append(iteration_result)

        print(iteration_str.format(j, k, current_l_w, momentum, mae, rmse))

        with open('{}_{}.json'.format(config_name, name), 'wt') as res_output:
            res_output.write(json.dumps(config_result, indent=4))

if __name__ == "__main__":
    args = parse_args()
    experiments = read_experiment(args.experiment)

    for experiment in experiments:
        name = experiment['name']
        train_path = experiment['train_path']
        test_path = experiment['test_path']
        sep = experiment['sep']
        configs = experiment['configs']

        all_users, all_movies, tests = load_dataset(train_path, test_path,
                                                    sep, user_based=True)

        for config in configs:
            run(name, train_path, config, all_users, all_movies, tests,
                None, sep, device=args.device)
