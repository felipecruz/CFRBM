run:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python -m cfrbm.user_based ubased.json

test:
	python -m unittest discover -v
