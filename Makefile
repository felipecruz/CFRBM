run:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cfrbm/user_based.py ubased.json

test:
	python -m unittest discover -v
