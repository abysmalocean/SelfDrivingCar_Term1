"""
Install the TensorFlow throw Anconda
conda create --name=IntroToTensorFlow python=3 anaconda
source activate IntroToTensorFlow
conda install -c conda-forge tensorflow

"""
# tru to use the tensorflow

import tensorflow as tf

# Create TensorFlow object called tensor
hello_constant = tf.constant('Hello World!')
A = tf.constant(1234)
B = tf.constant([12, 34,5,6,78])
C = tf.constant([ [123,456,789], [222,333,444] ])
x = tf.placeholder(tf.string)

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    output = sess.run(hello_constant)
    print(output)
    output2 = sess.run(A)
    print(output2)
    output3 = sess.run(x,feed_dict={x:"Hello world Tensor Flow"})
    print(output3)
