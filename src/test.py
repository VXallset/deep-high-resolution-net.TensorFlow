"""
This file is used to test the model using the AI Challenger dataset.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 27, 2019

"""
import numpy as np
import tensorflow as tf
from hrnet import *
import dataset
from heatmap import *
import time
import os

from functools import reduce
from operator import mul

def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


def main(device_option='/gpu:0'):
    batch_size = 1
    num_epochs = 10
    image_numbers = 378352
    #image_numbers = 4500

    root_path = os.getcwd()[:-3]

    datasetname = os.path.join(root_path, 'dataset/train.tfrecords')
    model_folder = os.path.join(root_path, 'models/')
    modelfile = os.path.join(root_path, 'models/epoch2.ckpt-567528')

    global_step = tf.Variable(0, trainable=False)

    image_name, image, keypoints_ver, keypoints_hor, keypoints_class = dataset.input_batch(
        datasetname=datasetname, batch_size=batch_size, num_epochs=num_epochs)

    input_images = tf.placeholder(tf.float32, [None, 256, 192, 3])
    ground_truth = tf.placeholder(tf.float32, [None, 256, 192, 14])

    input_images = tf.cast(input_images / 255.0, tf.float32, name='change_type')
    net_output = HRNet(input=input_images)
    loss = compute_loss(net_output=net_output, ground_truth=ground_truth)

    saver = tf.train.Saver()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''

    with tf.Session() as sess:
        with tf.device(device_option):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path=modelfile)
            #print(get_num_params())

            writer = tf.summary.FileWriter('../log/', sess.graph)
            start_time = time.time()
            try:
                distances = 0
                classes = 0
                for step in range(int(image_numbers / batch_size)):
                    _img, _kp_ver, _kp_hor, _kp_class = sess.run(
                        [image, keypoints_ver, keypoints_hor, keypoints_class])
                    _gt = calculate_groundtruth_heatmap(_kp_ver, _kp_hor, _kp_class)

                    tloss, tnet_output = sess.run([loss, net_output],
                                                  feed_dict={input_images: _img, ground_truth: _gt})

                    prediction = decode_output(tnet_output, threshold=0.001)
                    gt_all = np.zeros((batch_size, 14*3))
                    gt_all[:, ::3] = _kp_ver
                    gt_all[:, 1::3] = _kp_hor
                    gt_all[:, 2::3] = _kp_class
                    distance = calculate_distance(prediction, gt_all)

                    if step == 0:
                        distances = distance
                        classes = _kp_class
                    elif step == 1000:
                        np.save('distances.npy', distances)
                        np.save('classes.npy', classes)
                        break
                    else:
                        distances = np.append(distances, distance, axis=0)
                        classes = np.append(classes, _kp_class, axis=0)

                    timgs = decode_pose(_img, tnet_output, threshold=0.001)
                    for i in range(batch_size):
                        io.imsave('../test_img/step{}_i_{}.jpg'.format(step, i), timgs[i])
                    print('Step = {:>6}/{:>6}, loss = {:.6f}, time = {}'
                          .format(step, int(image_numbers / batch_size), tloss,
                                  time.time() - start_time))
                    print('---------------------------------------------------------------------------------')

            except tf.errors.OutOfRangeError:
                print('End testing...')
            finally:
                total_time = time.time() - start_time
                print('Running time: {} s'.format(total_time))
                print('Done!')


if __name__ == '__main__':
    main(device_option='/gpu:0')
