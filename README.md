# CFRBM

This is an implementation of the RBM model for the collaborative filtering task.

## Limitation

This code works, so far, only with ratings from 1 to 5.

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

# License

Mit. See `LICENSE` file

# Informal references

* http://stats.stackexchange.com/questions/191918/predicting-with-restricted-boltzmann-machines-for-collaborative-filtering/192829#192829


## References

* Ruslan Salakhutdinov, Andriy Mnih e Geoffrey Hinton. “Restricted Boltzmann machines for collaborative filtering”. Em: In Machine Learning, Proceedings of the Twenty-fourth International Conference (ICML 2004). ACM. AAAI Press, 2007, pp. 791–798.
* Yun Zhu, Yanqing Zhang e Yi Pan. “Large-scale restricted boltzmann machines on single GPU.” Em: BigData Conference. Ed. por Xiaohua Hu et al. IEEE, 2013, pp. 169–174. isbn: 978-1-4799-1292-6. url: http: / / dblp . uni - trier . de / db / conf / bigdataconf / bigdataconf2013 . html#ZhuZP13.
* Geoffrey E. Hinton. “A Practical Guide to Training Restricted Boltzmann Machines”. Em: Neural Networks: Tricks of the Trade - Second Edition. 2012, pp. 599–619. doi: 10.1007/978-3-642-35289- 8_32. url: http://dx.doi.org/10.1007/978-3-642-35289-8_32.
* Kostadin Georgiev e Preslav Nakov. “A non-IID Framework for Collaborative Filtering with Restricted Boltzmann Machines”. Em: Proceedings of the 30th International Conference on Machine Learning, Cycle 3. Vol. 28. JMLR Proceedings. JMLR.org, 2013, pp. 1148–1156.
