"""Shared training and evaluation for user- and item-oriented profiles."""

import json
import sys
import time

from collections import defaultdict
from datetime import datetime
from math import sqrt

import numpy as np
import torch

from .rbm import CFRBM
from .device import parse_args
from .experiments import read_experiment
from .utils import revert_expected_value, iteration_str
from .dataset import load_dataset
from .preparation import PreparedProfiles


def log(message):
    print(f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}] {message}")


def synchronize(device):
    # Accelerator kernels run asynchronously; wait for them before timing.
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def run(name, dataset, config, all_users, all_movies, tests, initial_v, sep,
        device=None, *, orientation='user'):
    """Train one config; tests must be grouped by the selected orientation.

    initial_v is retained for compatibility with the original runner API.
    """
    if orientation not in ('user', 'item'):
        raise ValueError("orientation must be 'user' or 'item'")
    visible_ids = all_movies if orientation == 'user' else all_users
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

    rbm = CFRBM(len(visible_ids) * 5, number_hidden, device=device)
    log(f"Using device: {rbm.device}")

    started = time.perf_counter()
    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for line in data:
            uid, mid, rat, _ = line.strip().split(sep)
            profile_id, visible_id = (uid, mid) if orientation == 'user' else (mid, uid)
            profiles[profile_id].append((visible_id, float(rat)))

    prepared = PreparedProfiles(profiles, visible_ids, tests)
    del profiles

    log(f"Users and ratings loaded in {time.perf_counter() - started:.2f}s")

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

        synchronize(rbm.device)
        train_started = time.perf_counter()
        train_rows = 0
        train_ratings = 0
        for train_batch, train_masks in prepared.training_batches(j, batch_size):
            train_rows += len(train_batch)
            # Each rating is a single one-hot unit, so the batch sum counts them.
            train_ratings += int(train_batch.sum())
            train(torch.from_numpy(train_batch).to(rbm.device),
                  torch.from_numpy(train_masks).to(rbm.device))
            sys.stdout.write('.')
            sys.stdout.flush()

        synchronize(rbm.device)
        train_seconds = time.perf_counter() - train_started

        eval_started = time.perf_counter()
        ratings = []
        predictions = []

        for batch, test_batch in prepared.test_batches(batch_size):
            positions = {profile_id: pos for pos, profile_id in enumerate(batch)}
            profile_predictions = revert_expected_value(
                predict(torch.from_numpy(test_batch).to(rbm.device)).cpu().numpy())
            for profile_id in batch:
                test_ratings = tests[profile_id]
                try:
                    for visible_id, rating in test_ratings:
                        current_profile = profile_predictions[positions[profile_id]]
                        predicted = current_profile[prepared.visible_index[visible_id]]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        eval_seconds = time.perf_counter() - eval_started

        distances = np.array(ratings) - np.array(predictions)

        mae = np.abs(distances).mean()
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

        print()
        log(iteration_str.format(j, k, current_l_w, momentum, mae, rmse).lstrip('\n'))
        train_seconds = max(train_seconds, 1e-9)
        eval_seconds = max(eval_seconds, 1e-9)
        log(f"Epoch {j} timing - train {train_seconds:.2f}s "
            f"({train_rows / train_seconds:,.0f} records/s, "
            f"{train_ratings / train_seconds:,.0f} ratings/s) - "
            f"eval {eval_seconds:.2f}s ({len(ratings) / eval_seconds:,.0f} ratings/s)")

        with open('{}_{}.json'.format(config_name, name), 'wt') as res_output:
            res_output.write(json.dumps(config_result, indent=4))


def main(argv=None, *, orientation='user'):
    args = parse_args(argv, orientation=orientation)
    experiments = read_experiment(args.experiment)

    for experiment in experiments:
        name = experiment['name']
        train_path = experiment['train_path']
        test_path = experiment['test_path']
        sep = experiment['sep']
        configs = experiment['configs']

        all_users, all_movies, tests = load_dataset(train_path, test_path,
                                                    sep, user_based=args.orientation == 'user')

        for config in configs:
            run(name, train_path, config, all_users, all_movies, tests,
                None, sep, device=args.device, orientation=args.orientation)


if __name__ == "__main__":
    main()
