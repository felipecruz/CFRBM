# CFRBM

This is an implementation of the RBM model for the collaborative filtering task.

## A short story

CFRBM started in July 2016, when deep learning in Python meant
[Theano](https://github.com/Theano/Theano): you described a symbolic graph,
compiled it with `theano.function`, and hoped `THEANO_FLAGS=device=gpu` would
find your CUDA card. Restricted Boltzmann Machines for collaborative filtering,
the model from Salakhutdinov, Mnih and Hinton's Netflix Prize paper, had few
open implementations at the time. This repository was one of the first: user-
and item-based RBMs with softmax-style rating units, contrastive divergence
written out by hand, and a GPU path for MovieLens experiments.

Then Theano was discontinued, Python 2 reached end of life, and the code
stopped running on any modern machine.

In 2026 the project was brought back with LLM coding agents. The work followed
a versioned plan in [`ROADMAP.md`](ROADMAP.md), carried out one item at a time,
with each step tested and checked against measurements:

1. **Python 3 port.** `xrange`, integer division and the old import style were
   fixed without touching the model.
2. **Locking in the math.** A NumPy reference implementation of the CD update
   was written first, so the new backend could be checked against it on fixed
   inputs.
3. **PyTorch backend.** The Theano graph became plain PyTorch tensors, with
   automatic selection of CUDA, Apple MPS or CPU.
4. **Fixing old bugs.** With parity established, the original training quirks
   were corrected: momentum now carries the real update step, the
   negative-phase gradient is masked correctly, and visible units use a true
   softmax with categorical sampling.
5. **Speed.** Rating profiles are now built once instead of every epoch with
   Python list searches. Epochs went from about 8 seconds to well under one on
   a laptop CPU.

The original references are kept below. Validation runs for each step are
recorded under [`validation/`](validation).

## Quick start (notebook)

[`notebooks/cfrbm_movielens.ipynb`](notebooks/cfrbm_movielens.ipynb) installs
this package from GitHub, downloads MovieLens 100k, trains the user-based model
and plots test MAE/RMSE for each epoch. The log shows a timestamp, records per
second and ratings per second for every epoch. You can also install the package
directly:

```sh
pip install "cfrbm @ git+https://github.com/felipecruz/CFRBM.git@main"
```

## Limitation

This code works, so far, only with ratings from 1 to 5.

## Dependencies

This library uses Python 3, NumPy and PyTorch. It has been tested with Python
3.12. From the repository root, create and activate a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Test dataset

Download and extract the `ml-100k` dataset in the repository root to run the
default experiment. The configuration expects `ml-100k/u1.base` and
`ml-100k/u1.test`:

```sh
curl -fLO https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

## Running

```sh
make run
```

or 

```sh
python -m cfrbm.runner ubased.json --orientation user
```

Run these commands from the repository root. For item-based experiments, use
`python -m cfrbm.runner ubased.json --orientation item`. The orientation defaults
to `user`; `make run ORIENTATION=item` selects item mode with Make. Both modes use the configuration's `batch_size` for training and
evaluation. The previous `cfrbm.user_based` and `cfrbm.item_based` module commands
and Python `run(...)` functions remain available as compatibility wrappers.
Item-based configurations with a batch size other than 10 now use that size,
which can change their training results.

The shared runner automatically selects CUDA, then Apple MPS, then CPU. To override
this choice, append `--device cpu`, `--device mps`, or `--device cuda:0` to the
module command, or use `make run DEVICE=cpu`. An unavailable device produces
an error. To use the virtual environment without activating it, run
`make run PYTHON=.venv/bin/python`.

The default experiment runs for 100 epochs and writes per-epoch MAE/RMSE to
`initial_experiment_100k-u1xu1.json` in the working directory. Repeating the
experiment overwrites that results file.

Run the tests with `make test` (or `make test PYTHON=.venv/bin/python`).

# License

Mit. See `LICENSE` file

# Informal references

* http://stats.stackexchange.com/questions/191918/predicting-with-restricted-boltzmann-machines-for-collaborative-filtering/192829#192829


## References

* Ruslan Salakhutdinov, Andriy Mnih e Geoffrey Hinton. “Restricted Boltzmann machines for collaborative filtering”. Em: In Machine Learning, Proceedings of the Twenty-fourth International Conference (ICML 2004). ACM. AAAI Press, 2007, pp. 791–798.
* Yun Zhu, Yanqing Zhang e Yi Pan. “Large-scale restricted boltzmann machines on single GPU.” Em: BigData Conference. Ed. por Xiaohua Hu et al. IEEE, 2013, pp. 169–174. isbn: 978-1-4799-1292-6. url: http: / / dblp . uni - trier . de / db / conf / bigdataconf / bigdataconf2013 . html#ZhuZP13.
* Geoffrey E. Hinton. “A Practical Guide to Training Restricted Boltzmann Machines”. Em: Neural Networks: Tricks of the Trade - Second Edition. 2012, pp. 599–619. doi: 10.1007/978-3-642-35289- 8_32. url: http://dx.doi.org/10.1007/978-3-642-35289-8_32.
* Kostadin Georgiev e Preslav Nakov. “A non-IID Framework for Collaborative Filtering with Restricted Boltzmann Machines”. Em: Proceedings of the 30th International Conference on Machine Learning, Cycle 3. Vol. 28. JMLR Proceedings. JMLR.org, 2013, pp. 1148–1156.
