import numpy as np
import theano
import theano.tensor as T
import lasagne

def build_mlp(ndims, nclasses, input_var=None):
	lin = lasagne.layers.InputLayer(shape=(None, 1, ndims))
	lin_drop = lasagne.layers.DropoutLayer(lin, p=0.2)
	lhid1 = lasagne.layers.DenseLayer(lin_drop, num_units=200, nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform())
	lout = lasagne.layers.DenseLayer(lhid1, num_units=nclasses, nonlinearity=lasagne.nonlinearities.softmax)
	return lout

def iterate_minibatches(inputs, targets, batchsize):
	for start in range(0, len(inputs) - batchsize + 1, batchsize):
		excerpt = indices[start:start+batchsize]
		yield inputs[excerpt], targets[excerpt]

def do_mlp(x, y, ndims, nclasses, n):
	input_var = T.tensor3('inputs')
	target_var = T.ivector('targets')
	network = build_mlp(ndims, nclasses, input_var)
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
	test_loss = test_loss.mean()
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)
	train_fn = theano.function([input_var, target_var], loss, updates=updates)
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

	print("training...")
	nepochs = 5
	for epoch in range(nepochs):
		train_err = 0
		train_batches = 0
		for batch in iterate_minibatches(x, y, 500):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1
		# Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
