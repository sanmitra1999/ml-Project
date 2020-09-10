import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import tensorflow as tf
import os 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit


lis = os.listdir('./dataset/')

X = []
y = []
def char2int(c):
	return [ord(c)-ord('A')]

#generates a weight variable of a given shape.
def weight_variable(shape):
  initial = tf.random.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
#generates a bias variable of a given shape.
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
#returns a 2d convolution layer with full stride
def conv_2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#down samples a feature map by 2X
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def angle(im,theta):
	return im

IMG_HEIGHT = 80
IMG_WIDTH = 80

for i in lis:
	if i[0]=='.':
		continue
	image = cv2.imread('./dataset/'+i,0)
	im = cv2.resize(image,(IMG_HEIGHT,IMG_WIDTH))
	# l,r=image.shape
	# l=IMG_HEIGHT-l
	# r=IMG_WIDTH-r
	# a=l/2
	# b=l-a
	# c=r/2
	# d=r-c
	# im =cv2.copyMakeBorder(image.copy(),a,b,c,d,cv2.BORDER_CONSTANT,value=0)
	X.append(np.reshape(im,(IMG_HEIGHT,IMG_WIDTH,1)))
	y.append(char2int(i[-5]))

y = OneHotEncoder().fit_transform(y).todense()
X = np.array(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
batch_size = 32
epochs = 15
prob = 0.25
tf.compat.v1.disable_eager_execution()
tf_data = tf.compat.v1.placeholder(tf.float32, shape=(None, IMG_HEIGHT, IMG_WIDTH,1))
tf_labels = tf.compat.v1.placeholder(tf.float32, shape=(None, 26))
keep_prob = tf.compat.v1.placeholder(tf.float32)

w1 = weight_variable([4, 4,1,8])
b1 = bias_variable([8])
layer_conv1 = tf.nn.relu(conv_2d(tf_data, w1) + b1)
layer1 = tf.nn.dropout(layer_conv1, rate=keep_prob)
# Pooling layer - downsamples by 2X.
layer_pool1 = max_pool_2x2(layer1)

# Second convolutional layer -- maps 32 feature maps to 64.
w2 = weight_variable([3, 3, 8, 16])
b2 = bias_variable([16])
layer_conv2 = tf.nn.relu(conv_2d(layer_pool1, w2) + b2)
layer2 = tf.nn.dropout(layer_conv2, rate=keep_prob)
# Second pooling layer.
layer_pool2 = max_pool_2x2(layer2)
wfc1 = weight_variable([(IMG_HEIGHT/4)*(IMG_WIDTH/4)*16, 256])
bfc1 = bias_variable([256])
flatten_pool2 = tf.reshape(layer_pool2, [-1,(IMG_HEIGHT/4)*(IMG_WIDTH/4)*16])
layer_fc1 = tf.nn.relu(tf.matmul(flatten_pool2, wfc1) + bfc1)
layer = tf.nn.dropout(layer_fc1, rate=keep_prob)

wfc2 = weight_variable([256, 26])
bfc2 = bias_variable([26])
y_pred = (tf.matmul(layer, wfc2) + bfc2)


labels = tf.stop_gradient(tf_labels)

loss=tf.reduce_mean(tf.compat.v2.nn.softmax_cross_entropy_with_logits(logits=y_pred,labels=labels))
tf_pred = tf.nn.softmax(y_pred)
tf_acc = 100*tf.reduce_mean(tf.compat.v1.to_float(tf.equal(tf.argmax(tf_pred, 1), tf.argmax(tf_labels, 1))))
params = tf.compat.v1.trainable_variables()
gradients = tf.gradients(loss, params)
optimizer = tf.compat.v1.train.RMSPropOptimizer(0.0005)
update_step = optimizer.apply_gradients(zip(gradients, params))
all_saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables())

with tf.compat.v1.Session() as sess:
	sess.run(tf.compat.v1.global_variables_initializer())
	# all_saver.restore(sess, '/data/shrey/copynet/model' + '/data-all')
	for epoch in xrange(2,epochs+1):
		# print 'alpha'
		epoch_loss = 0
		ss = ShuffleSplit(n_splits=X_train.shape[0]/batch_size, train_size=batch_size,random_state=42)
		ss.get_n_splits(X_train, y_train)
		for step, (idx, _) in enumerate(ss.split(X_train,y_train), start=1):
			train = X_train[idx]
			_,co = sess.run([update_step,loss],feed_dict={tf_data:train, tf_labels:y_train[idx],keep_prob:prob})
			epoch_loss+=co
			if step%10 == 0:
				valid_accuracy = sess.run(tf_acc,feed_dict={tf_data:X_test, tf_labels:y_test,keep_prob:0.0})
				print('Step %i \t Valid. Acc. = %f \n'%(step, valid_accuracy))
		print("Epoch",epoch,'completed out of',epochs,'loss:',epoch_loss)
		all_saver.save(sess, '/Users/shrey/Desktop/model/')




