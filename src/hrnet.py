"""
This is the structure of the HRNet-32, an implementation of the CVPR 2019 paper "Deep High-Resolution Representation
Learning for Human Pose Estimation" using TensorFlow.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 06, 2019

"""
import tensorflow as tf
from utils import *


def stage1(input, name='stage1', is_training=True):
    output = []
    with tf.variable_scope(name):
        s1_res1 = residual_unit_bottleneck(input, name='rs1', is_training=is_training)
        s1_res2 = residual_unit_bottleneck(s1_res1, name='rs2', is_training=is_training)
        s1_res3 = residual_unit_bottleneck(s1_res2, name='rs3', is_training=is_training)
        s1_res4 = residual_unit_bottleneck(s1_res3, name='rs4', is_training=is_training)
        output.append(conv_2d(s1_res4, channels=32, activation=leaky_Relu, name=name + '_output',
                              is_training=is_training))
    return output


def stage2(input, name='stage2', is_training=True):
    with tf.variable_scope(name):
        sub_networks = exchange_between_stage(input, name='between_stage', is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block', is_training=is_training)
    return sub_networks


def stage3(input, name='stage3', is_training=True):
    with tf.variable_scope(name):
        sub_networks = exchange_between_stage(input, name=name, is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block1', is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block2', is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block3', is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block4', is_training=is_training)
    return sub_networks


def stage4(input, name='stage4', is_training=True):
    with tf.variable_scope(name):
        sub_networks = exchange_between_stage(input, name=name, is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block1', is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block2', is_training=is_training)
        sub_networks = exchange_block(sub_networks, name='exchange_block3', is_training=is_training)
    return sub_networks


def HRNet(input, is_training=True, eps=1e-10):
    output = stage1(input=input, is_training=is_training)
    output = stage2(input=output, is_training=is_training)
    output = stage3(input=output, is_training=is_training)
    output = stage4(input=output, is_training=is_training)

    # The output contains 4 sub-networks, we only need the first one, which contains information of all
    # resolution levels
    output = output[0]

    # using a 3x3 convolution to reduce the channels of feature maps to 14 (the number of keypoints)
    output = conv_2d(output, channels=14, kernel_size=3, batch_normalization=False, name='change_channel',
                     is_training=is_training, activation=tf.nn.relu)
    # sigmoid can convert the output to the interval of (0, 1)
    # output = tf.nn.sigmoid(output, name='net_output')

    # If we don't normalize the value of the output to 1, the net may predict the values on all pixels to be 0, which
    # will make the loss of one image to be around 1.75 (batch_size = 1, 256, 192, 14). This is because that the value
    # of an 3 x 3 gaussian kernel is g =
    # [[0.07511361 0.1238414  0.07511361]
    #  [0.1238414  0.20417996 0.1238414 ]
    #  [0.07511361 0.1238414  0.07511361]]

    # so g^2 =
    # [[0.00564205 0.01533669 0.00564205]
    #  [0.01533669 0.04168945 0.01533669]
    #  [0.00564205 0.01533669 0.00564205]]
    # therefore, np.sum(g^2) * 14 = 1.75846

    # In order to avoid this from happening, we need to normalize the value of the net output by dividing the value on
    # all pixels by the sum of the value on that image (1, 256, 192, 1). Or we may calculate the classification loss
    # to indicate the class of the key points.


    # sum up the value on each pixels, the result should be a [batch_size, 14] tensor, then expend dim to be
    # [batch_size, 1, 1, 14] tensor so as to normalize the output
    output_sum = tf.expand_dims(tf.expand_dims(tf.reduce_sum(tf.reduce_sum(output, axis=-2),
                                                             axis=-2), axis=-2), axis=-2, name='net_output_sum')

    output = tf.truediv(output, output_sum + eps, name='net_output_final')

    return output


def mytest():
    input = tf.ones((16, 256, 192, 3))
    output = HRNet(input)

    print(output)


def compute_loss(net_output, ground_truth):
    diff = tf.square(tf.subtract(net_output, ground_truth), name='square_difference')
    loss = tf.reduce_sum(diff, name='loss')
    #loss = tf.losses.mean_squared_error(ground_truth, net_output)

    return loss


if __name__ == '__main__':
    mytest()
