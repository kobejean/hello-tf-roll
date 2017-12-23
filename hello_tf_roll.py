import tensorflow as tf
manip = tf.load_op_library('user_ops/roll_op.so')

tensor = tf.constant([[1, 2, 3, 4, 5],
                      [6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15]])
rolled = manip.custom_roll(tensor, shift=[2,-3], axis=[0,1])
# rolled = tf.manip.roll(tensor, shift=[2,-3], axis=[0,1])
sess = tf.Session()
print(rolled)
print("Before:", sess.run(tensor))
print("After:", sess.run(rolled))
