# CFRBM

This is an implementation of the RBM model for the collaborative filtering task.

## Dependencies

This library relies on Theano.

```
$ pip install theano
```

## Test dataset

You must download the `ml-100k` dataset to run the default experiment.

```
$ wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
```

## Running

```
$ make run
```

or 

```
$ THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cfrbm/user_based.py ubased.json
```
