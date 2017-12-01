#ML-Tensorflow-NeuralNetwork.py
'''
input -> weight -> haiiden layer1 (activation function)
-> weights -> hidden layer2 (activation function) -> output layer

compare output to intended output -> cost function (cross entropy)
optimization function (optimizer) -> minimize cost (AdamOptimizer ... SGD, AdaGrad)

backpropagation

feed forward _ backprop = epoch
'''
import time
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data', one_hot = True)

n_nodes_l1 = 500
n_nodes_l2 = 500
n_nodes_l3 = 15

n_classes = 10
#batch is how many data for input once
#batch_size means 10 times for data to input
batch_size = 10

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_l1])),
						'biases':tf.Variable(tf.random_normal([n_nodes_l1]))}
	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_l1, n_nodes_l2])),
						'biases':tf.Variable(tf.random_normal([n_nodes_l2]))}
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_l2, n_nodes_l3])),
						'biases':tf.Variable(tf.random_normal([n_nodes_l3]))}
	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_l3, n_classes])),
						'biases':tf.Variable(tf.random_normal([n_classes]))}

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)
	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)
	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)
	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, 
		labels = y))
	#                 default 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, each_batch_cost = sess.run([optimizer, cost], 
					feed_dict = {x : epoch_x, y : epoch_y})
				epoch_loss += each_batch_cost 

			print('epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		print(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x: mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
