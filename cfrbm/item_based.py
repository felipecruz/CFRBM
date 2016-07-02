import json
import sys

from collections import defaultdict
from math import sqrt

import numpy as np
import theano.tensor as T

import utils

from rbm import CFRBM
from experiments import read_experiment
from utils import revert_expected_value, expand, iteration_str
from dataset import load_dataset



def run(name, dataset, config, all_users, all_movies, tests, initial_v, sep):
    config_name = config['name']
    number_hidden = config['number_hidden']
    epochs = config['epochs']
    ks = config['ks']
    momentums = config['momentums']
    l_w = config['l_w']
    l_v = config['l_v']
    l_h = config['l_h']
    decay = config['decay']

    config_result = config.copy()
    config_result['results'] = []

    vis = T.matrix()
    vmasks = T.matrix()

    rbm = CFRBM(len(all_users) * 5, number_hidden)

    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            profiles[mid].append((uid, float(rat)))

    print("Users and ratings loaded")

    for j in range(epochs):
        def get_index(col):
            if j/(epochs/len(col)) < len(col):
                return j/(epochs/len(col))
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

        train = rbm.cdk_fun(vis,
                            vmasks,
                            k=k,
                            w_lr=current_l_w,
                            v_lr=current_l_v,
                            h_lr=current_l_h,
                            decay=decay,
                            momentum=momentum)
        predict = rbm.predict(vis)

        batch_size = 10
        for batch_i, batch in enumerate(utils.chunker(profiles.keys(),
                                                      batch_size)):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            masks = {}
            for movieid in batch:
                movie_profile = [0.] * len(all_users)
                mask = [0] * (len(all_users) * 5)

                for user_id, rat in profiles[movieid]:
                    movie_profile[all_users.index(user_id)] = rat
                    for _i in range(5):
                        mask[5 * all_users.index(user_id) + _i] = 1

                example = expand(np.array([movie_profile])).astype('float32')
                bin_profiles[movieid] = example
                masks[movieid] = mask

            movies_batch = [bin_profiles[id] for id in batch]
            masks_batch = [masks[id] for id in batch]
            train_batch = np.array(movies_batch).reshape(size,
                                                         len(all_users) * 5)
            train_masks = np.array(masks_batch).reshape(size,
                                                        len(all_users) * 5)
            train_masks = train_masks.astype('float32')
            train(train_batch, train_masks)
            sys.stdout.write('.')
            sys.stdout.flush()

        batch_size = 10
        ratings = []
        predictions = []

        for batch in utils.chunker(tests.keys(), batch_size):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            masks = {}
            for movieid in batch:
                movie_profile = [0.] * len(all_users)
                mask = [0] * (len(all_users) * 5)

                for userid, rat in profiles[movieid]:
                    movie_profile[all_users.index(userid)] = rat
                    for _i in range(5):
                        mask[5 * all_users.index(userid) + _i] = 1

                example = expand(np.array([movie_profile])).astype('float32')
                bin_profiles[movieid] = example
                masks[movieid] = mask

            positions = {movie_id: pos for pos, movie_id in enumerate(batch)}
            movies_batch = [bin_profiles[el] for el in batch]
            test_batch = np.array(movies_batch).reshape(size,
                                                        len(all_users) * 5)
            movie_predictions = revert_expected_value(predict(test_batch))
            for movie_id in batch:
                test_users = tests[movie_id]
                try:
                    for user, rating in test_users:
                        current_movie = movie_predictions[positions[movie_id]]
                        predicted = current_movie[all_users.index(user)]
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

    experiment = read_experiment(sys.argv[1])

    name = experiment['name']
    train_path = experiment['train_path']
    test_path = experiment['test_path']
    sep = experiment['sep']

    all_users, all_movies, tests = load_dataset(train_path, test_path, sep,
                                                user_based=False)

    for config in experiment['configs']:
        run(name, train_path, config, all_users, all_movies, tests, None, sep)
