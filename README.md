# CFRBM

This is an implementation of the RBM model for the collaborative filtering task.

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
python -m cfrbm.user_based ubased.json
```

Run these commands from the repository root. For item-based experiments, use
`python -m cfrbm.item_based ubased.json`.

Both runners automatically select CUDA, then Apple MPS, then CPU. To override
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
