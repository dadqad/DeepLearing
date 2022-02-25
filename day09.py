# Auto Encodeing 模型
#代码参考了 莫凡学 Tensorflow 代码，有兴趣的可以学习

import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import psutil
import os
info = psutil.virtual_memory()
#   
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)          #导入数据
print(tf.test.is_gpu_available())

learning_rate = 0.001
training_epochs = 200
batch_size = 256
display_size = 1
examples_to_show = 10
display_step = 10

n_input = 784

X = tf.placeholder('float',[None,n_input])
# y = tf.placeholder('float',[None,10])

n_hidden_1 =256
n_hidden_2 =128

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),

    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_input])),  
}

biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2])),

    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2':tf.Variable(tf.random_normal([n_input])),   
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op

y_true = X

cost = tf.reduce_mean(tf.pow(y_true-y_pred,2))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.initialize_all_variables()        
# plt.figure()
plt.ion()        
f,a= plt.subplots(2,10,figsize=(10,2))

with tf.Session() as sess:
    sess.run(init)

    total_batch = int(mnist.train.num_examples/batch_size)

    for epoach in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={X:batch_xs})
            del batch_xs
            del batch_ys

            # if(epoach % display_step == 0):
            #     print("Epoach",'%04d'%(epoach+1),"cost=","{:.9f}",format(c))
        print("Optimization Finished!")
        plt.suptitle(f"model:AutoEncoder epoch: {epoach} loss: {c}", ha='center')
        encode_decode =sess.run(y_pred,feed_dict={X:mnist.test.images[10:10+examples_to_show]})
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[10+i],(28,28)))
            a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
        # plt.show()
        plt.pause(0.01)
        print(u'内存使用：',psutil.Process(os.getpid()).memory_info().rss)
        del encode_decode
        del c
        # plt.close('all')
