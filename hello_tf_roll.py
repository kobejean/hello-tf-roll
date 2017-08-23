import tensorflow as tf
tensor = tf.constant([1, 2, 3, 4, 5],
                     [6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15])
rolled = tf.roll(tensor, shift=[2,-3], axis=[0,1])
sess = tf.Session()
print(sess.run(rolled))
