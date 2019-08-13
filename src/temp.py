"""
This file is used to test the model using the AI Challenger dataset.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Apr 13, 2019

"""
import numpy as np
import tensorflow as tf
from hrnet import *
import dataset
from heatmap import *
import time
import os
from skimage import io
from skimage.transform import resize
import cv2

def main(use_GPU = True):
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
    device = '/gpu:0'
    if not use_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = '/cpu:0'

    video_capture = cv2.VideoCapture('./cxk.mp4')
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    start_second = 0
    start_frame = fps * start_second
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with tf.Session() as sess:
        with tf.device(device):
            sess.run(tf.global_variables_initializer())
            saver.restore(sess=sess, save_path=modelfile)


            try:
                framid = 0
                while True:
                    start_time = time.time()
                    retval, img_data = video_capture.read()
                    if not retval:
                        break
                    img_data = cv2.cvtColor(img_data, code=cv2.COLOR_BGR2RGB)
                    _img = cv2.resize(img_data, (192, 256))
                    _img = np.array([_img])

                    tnet_output = sess.run(net_output, feed_dict={input_images: _img})

                    #prediction = decode_output(tnet_output, threshold=0.001)

                    timgs = decode_pose(_img, tnet_output, threshold=0.001)
                    resultimg = timgs[0]/ 255.0

                    io.imsave('../demo_img/frame_{}.jpg'.format(framid), resultimg)
                    framid += 1
                    print('time = {}'.format(time.time() - start_time))
                    print('---------------------------------------------------------------------------------')

            except tf.errors.OutOfRangeError:
                print('End testing...')
            finally:
                total_time = time.time() - start_time
                print('Running time: {} s'.format(total_time))
                print('Done!')


if __name__ == '__main__':
    main(use_GPU=True)
