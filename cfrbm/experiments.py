import json


def read_experiment(path):

    with open(path) as data:
        experiments = json.loads(data.read())['experiments']

    return experiments
