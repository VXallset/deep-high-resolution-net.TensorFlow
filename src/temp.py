"""
This file is used to test the model using the AI Challenger dataset.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 27, 2019

"""
import numpy as np
import tensorflow as tf
from src.hrnet import *
from src import dataset
from src.heatmap import *
import time
import os
from skimage import io
from skimage.transform import resize


def main(device_option='/gpu:0'):
    batch_size = 1
    num_epochs = 10
    image_numbers = 378352
    #image_numbers = 4500

    root_path = os.getcwd()[:-3]

    datasetname = os.path.join(root_path, 'dataset/test.tfrecords')
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
    _img = io.imread('../demo_img/cxk.jpg')
    _img = np.array(resize(_img, (256, 192, 3)) * 255, dtype=np.int)
    _img = np.reshape(_img, [1, 256, 192, 3])

    with tf.Session() as sess:
        with tf.device(device_option):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path=modelfile)

            start_time = time.time()
            try:
                tnet_output = sess.run(net_output, feed_dict={input_images: _img})

                prediction = decode_output(tnet_output, threshold=0.001)


                timgs = decode_pose(_img, tnet_output, threshold=0.001)
                resultimg = timgs[0]/ 255.0


                io.imsave('../demo_img/result.jpg', resultimg)
                print('time = {}'.format(time.time() - start_time))
                print('---------------------------------------------------------------------------------')

            except tf.errors.OutOfRangeError:
                print('End testing...')
            finally:
                total_time = time.time() - start_time
                print('Running time: {} s'.format(total_time))
                print('Done!')


if __name__ == '__main__':
    main(device_option='/gpu:0')

