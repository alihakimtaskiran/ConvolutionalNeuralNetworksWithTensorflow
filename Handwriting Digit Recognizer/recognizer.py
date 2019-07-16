import cv2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/mnist",one_hot=True,reshape=False)
	
X=tf.placeholder(tf.float32,[None,28,28,1])
Y=tf.placeholder(tf.float32,[None,10])

wc1=tf.Variable(tf.random.truncated_normal([6,6,1,16],stddev=0.2))
bc1=tf.Variable(tf.random.truncated_normal([16],stddev=0.2))

wc2=tf.Variable(tf.random.truncated_normal([5,5,16,32],stddev=0.2))
bc2=tf.Variable(tf.random.truncated_normal([32],stddev=0.2))

wd1=tf.Variable(tf.random.truncated_normal([1568,256],stddev=0.2))
bd1=tf.Variable(tf.random.truncated_normal([256],stddev=0.2))

wd2=tf.Variable(tf.random.truncated_normal([256,64],stddev=0.2))
bd2=tf.Variable(tf.random.truncated_normal([64],stddev=0.2))

wdo=tf.Variable(tf.random.truncated_normal([64,10],stddev=0.2))
bdo=tf.Variable(tf.random.truncated_normal([10],stddev=0.2))

y=tf.nn.relu(tf.nn.conv2d(X,wc1,strides=[1,1,1,1],padding="SAME")+bc1)
y=tf.nn.max_pool(y,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
y=tf.nn.relu(tf.nn.conv2d(y,wc2,strides=[1,1,1,1],padding="SAME")+bc2)
y=tf.nn.max_pool(y,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
y=tf.reshape(y,(-1,1568))
y=tf.nn.tanh(tf.linalg.matmul(y,wd1)+bd1)
y=tf.nn.tanh(tf.linalg.matmul(y,wd2)+bd2)
y_pred=tf.nn.softmax(tf.linalg.matmul(y,wdo)+bdo)

xent=-tf.reduce_sum(Y*tf.math.log(y_pred))

correct_pred=tf.equal(tf.argmax(Y,1),tf.argmax(y_pred,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "tmp/model.ckpt")
    print("Acc is",sess.run(accuracy,feed_dict={X:mnist.test.images[0:59],Y:mnist.test.labels[0:59]}))
    while(True):
        path=input("file:")
        resim=cv2.imread(path,0).astype(np.float32)/255
        resim=cv2.resize(resim,(28,28)).reshape(28,28,1)
        val_=sess.run(y_pred,feed_dict={X:(resim,)})
        val=sess.run(tf.argmax(y_pred,1),feed_dict={X:(resim,)})
        print("Resimdeki sayı=",val[0])
