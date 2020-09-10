import numpy as np
import tensorflow as tf


def char2int(c):
	return [ord(c)-ord('A')]
def weight_variable(shape):
  initial = tf.random.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def pred(chars):
	tf.compat.v1.disable_eager_execution()
	tf_data = tf.compat.v1.placeholder(tf.float32, shape=(None, 80, 80,1))
	keep_prob = tf.compat.v1.placeholder(tf.float32)
	w1 = weight_variable([4, 4,1,8])
	b1 = bias_variable([8])
	layer_conv1 = tf.nn.relu(conv_2d(tf_data, w1) + b1)
	layer1 = tf.nn.dropout(layer_conv1, rate=keep_prob)
	layer_pool1 = max_pool_2x2(layer1)

	
	w2 = weight_variable([3, 3, 8, 16])
	b2 = bias_variable([16])
	layer_conv2 = tf.nn.relu(conv_2d(layer_pool1, w2) + b2)
	layer2 = tf.nn.dropout(layer_conv2, rate=keep_prob)
	layer_pool2 = max_pool_2x2(layer2)
	wfc1 = weight_variable([20*20*16, 256])
	bfc1 = bias_variable([256])
	flatten_pool2 = tf.reshape(layer_pool2, [-1,20*20*16])
	layer_fc1 = tf.nn.relu(tf.matmul(flatten_pool2, wfc1) + bfc1)
	layer = tf.nn.dropout(layer_fc1, rate=keep_prob)

	wfc2 = weight_variable([256, 26])
	bfc2 = bias_variable([26])
	y_pred =tf.argmax((tf.matmul(layer, wfc2) + bfc2),1)
	all_saver = tf.compat.v1.train.Saver()

	ans = []
	with tf.compat.v1.Session() as sess:
		all_saver.restore(sess, 'model/')
		epoch_loss = 0
		for i in chars:
			train = np.array(i)
			a = sess.run(y_pred,feed_dict={tf_data:train,keep_prob:0.0})
			ans.append(list(a))
		return ans
		




