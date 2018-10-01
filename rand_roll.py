import tensorflow as tf

with tf.Session() as sess:
    tensor = tf.Variable([[1,2,3],[4,5,6]])
    init = tf.global_variables_initializer()
    sess.run(init)

    x_len = tensor.get_shape().as_list()[1]
    # random roll amount
    i = tf.random_uniform(shape=[1], maxval=x_len, dtype=tf.int32)
    with tf.device('/gpu:0'):
        output = tf.manip.roll(tensor, shift=i, axis=[1])

    for i in range(10):
        print(sess.run(output))
