import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import configparser
from minibatcher import MiniBatcher 


config=configparser.ConfigParser()
config.read('./config.ini')
print ('loading data...')
X_tr=np.load(config.get('DataPath','X_tr_path'))
y_tr=np.load(config.get('DataPath','y_tr_path'))
X_val=np.load(config.get('DataPath','X_val_path'))
y_val=np.load(config.get('DataPath','y_val_path'))

# Training Parameters
learning_rate = float(config.get('Parameters','learning_rate'))
batch_size = int(config.get('Parameters','batch_size'))
display_step = int(config.get("Parameters",'display_step'))
EPOCH=int(config.get('Parameters','EPOCH'))

minibatcher=MiniBatcher(batch_size,X_tr.shape[0])


tf.reset_default_graph()
# Network Parameters
num_blocks=int(config.get('NetworkParameters','num_blocks'))
num_cells=int(config.get('NetworkParameters','num_cells'))
num_input = int(config.get('NetworkParameters','num_input'))
timesteps = int(config.get('NetworkParameters','timesteps'))
num_hidden = int(config.get('NetworkParameters','num_hidden'))
num_classes = int(config.get('NetworkParameters','num_classes'))
PEEHOLE=config.getboolean('NetworkParameters','peehole')
# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

tf.reset_default_graph()
# Network Parameters
num_blocks=2
num_cells=2
num_input = 1 # MNIST data input (img shape: 28*28)
timesteps = 8 # timesteps
num_hidden = 15 # hidden layer num of features
num_classes = 2 # MNIST total classes (0-9 digits)
PEEHOLE=False
# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
	'out': tf.Variable(tf.random_normal([num_hidden*num_cells*num_blocks, num_classes]))
}
biases = {
	'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
	x = tf.unstack(x, timesteps, 1)
	blocks_output=[]
	# Define a lstm cell with tensorflow

	for i in range(num_blocks):
		with tf.variable_scope("block_"+str(i), reuse=tf.AUTO_REUSE):
			lstm_cel = rnn.LSTMCell(num_hidden*num_cells, forget_bias=1.0,use_peepholes=PEEHOLE)
			outputs, states = rnn.static_rnn(lstm_cel, x, dtype=tf.float32)
			blocks_output.append(outputs[-1])

	concat_output=tf.concat(blocks_output,1)
	return tf.matmul(concat_output, weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))

argmax_prediction = tf.argmax(prediction, 1)
argmax_y = tf.argmax(Y, 1)

TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
Total=TP+TN+FP+FN
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1= 2*(Recall * Precision) / (Recall + Precision)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

	# Run the initializer
	sess.run(init)
	for epoch in range(EPOCH):
		print ("*"*10,'EPOCH :',epoch,"*"*10)
		step=0
		for idxs in minibatcher.get_one_batch():
			batch_x, batch_y = X_tr[idxs],y_tr[idxs]
			# Reshape data to get 28 seq of 28 elements
			batch_x = batch_x.reshape((-1, timesteps, num_input))
			# Run optimization op (backprop)
			sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
			if step % display_step == 0 or step == 1:
				# Calculate batch loss and accuracy
				loss, acc,f1,recall = sess.run([loss_op, accuracy,F1,Recall], feed_dict={X: batch_x,
											Y: batch_y})
				print("Step " + str(step) + ", Minibatch Loss= " + \
						"{:.4f}".format(loss) + ", Training Accuracy= " + \
						"{:.3f}".format(acc) + ", F1= "+\
						"{:.3f}".format(f1)+", Recall= "+\
						"{:.3f}".format(recall)
						)
			step+=1
        
		acc_te,recall_te,f1_te=sess.run([accuracy,Recall,F1], feed_dict={X: X_val, Y: y_val})
		print("Testing Accuracy= " + \
					"{:.3f}".format(acc_te) + ", Recall= "+\
					"{:.3f}".format(recall_te)+", F1= "+\
					"{:.3f}".format(f1_te))

	print("Optimization Finished!")







