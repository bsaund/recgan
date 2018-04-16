import tensorflow as tf
import os
import numpy as np
import scipy.io


vox_input_dim = 64
vox_output_dim = 256


def conv3d(x, out_c, str, name, k=4, pad='SAME'):
    xavier_init = tf.contrib.layers.xavier_initializer()
    zero_init = tf.zeros_initializer()
    in_c = x.get_shape()[4]
    w = tf.get_variable(name + '_w', [k, k, k, in_c, out_c], initializer=xavier_init)
    b = tf.get_variable(name + '_b', [out_c], initializer=zero_init)
    stride = [1, str, str, str, 1]
    y = tf.nn.bias_add(tf.nn.conv3d(x, w, stride, pad), b)
    Ops.variable_sum(w, name)
    return y
    


class RecGan3D:
    def __init__(self):
        pass

    def createAutoEncoder():
        with tf.variable_scope('generator'), tf.device('/gpu:0'):
            # tf.reshape(self.x, [-1, vox_input_dim
            in32 = conv3d(self.x, 
            
        

    def createNetwork(self):
        self.x = tf.placeholder(shape=[None, vox_input_dim, vox_input_dim, vox_input_dim, 1], dtype=tf.float32)
        self.y_true = tf.placeholder(shape=[None, vox_output_dim, vox_output_dim, vox_output_dim, 1], dtype=tf.float32)

        

        self.y_pred = self.createAutoEncoder()

        # with tf.variable_scope('discriminator'):
        #     self.x


if __name__ == "__main__":
        print "hi"
        network = RecGan3D()
        network.createNetwork()
