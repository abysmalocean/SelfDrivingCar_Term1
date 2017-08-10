# Solution is available in the other "solution.py" tab
import tensorflow as tf

x = tf.constant(10)
y = tf.constant(2)
z = tf.subtract(tf.divide(x,y),tf.cast(tf.constant(1),tf.float64))

init = tf.global_variables_initializer()

n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
reduce_SUM = tf.log(tf.cast(([1, 2, 3, 4, 5]),tf.float64))


with tf.Session() as sess:
    output = sess.run(z)
    print(output)
    sess.run(init)
    print(sess.run(reduce_SUM))
