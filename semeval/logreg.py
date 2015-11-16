import numpy as np
import theano
import theano.tensor as T

def do_logreg_ova(train_x, train_y, ndims, nclasses, n):
	for i in range(nclasses):
		y = map(lambda label: 1 * (label == i), train_y)
		acc = do_logreg(train_x, y, ndims, nclasses, n, i)
		print("class %d: %.4f" % (i, acc))

def do_logreg(train_x, train_y, ndims, nclasses, n, target_class):
	train_steps = 1000
	x = T.matrix("x")
	y = T.vector("y")
	w = theano.shared(np.zeros(ndims), name="w")
	b = theano.shared(0., name="b")

	p1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
	prediction = p1 > 0.5
	xent = -y * T.log(p1) - (1 - y) * T.log(1 - p1)
	cost = xent.mean() + 0.01 * (w**2).sum()
	gw, gb = T.grad(cost, [w, b])

	train = theano.function(
		inputs=[x,y],
		outputs=[prediction, xent],
		updates=((w, w-0.01*gw), (b, b-0.01*gb))
		)
	predict = theano.function(inputs=[x], outputs=prediction)

	print("training...")
	for i in range(train_steps):
		pred, err = train(train_x, train_y)

	prediction = predict(train_x)
	ncorrect = 0.0
	total = 0.0
	for pred, ans in zip(prediction, train_y):
		if pred == ans:
			ncorrect += 1.0
		total += 1.0
	return ncorrect/total
