"""
This file is used to train the HRNet-32 model.

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


def main(gpu_divice='/gpu:0'):
    is_training = True

    batch_size = 1
    num_epochs = 10
    image_numbers = 378352
    learning_rate = 0.001
    save_epoch_number = 1
    root_path = os.getcwd()[:-3]

    datasetname = os.path.join(root_path, 'dataset/train.tfrecords')
    model_folder = os.path.join(root_path, 'models/')
    modelfile = os.path.join(root_path, 'models/model.ckpt')

    global_step = tf.Variable(0, trainable=False)

    image_name, image, keypoints_ver, keypoints_hor, keypoints_class = dataset.input_batch(
        datasetname=datasetname, batch_size=batch_size, num_epochs=num_epochs)

    input_images = tf.placeholder(tf.float32, [None, 256, 192, 3])
    ground_truth = tf.placeholder(tf.float32, [None, 256, 192, 14])

    input_images = tf.cast(input_images / 255.0, tf.float32, name='change_type')
    net_output = HRNet(input=input_images, is_training=is_training)
    loss = compute_loss(net_output=net_output, ground_truth=ground_truth)

    saver = tf.train.Saver()
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        with tf.device(gpu_divice):
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter('../log/', sess.graph)
            start_time = time.time()
            try:
                for epoch in range(num_epochs):
                    epoch_time = time.time()
                    for step in range(int(image_numbers / batch_size)):
                        _img, _kp_ver, _kp_hor, _kp_class = sess.run(
                            [image, keypoints_ver, keypoints_hor, keypoints_class])
                        _gt = calculate_groundtruth_heatmap(_kp_ver, _kp_hor, _kp_class)

                        train_step.run(feed_dict={input_images: _img, ground_truth: _gt})

                        if step % 100 == 0:
                            tloss, tnet_output = sess.run([loss, net_output],
                                                          feed_dict={input_images: _img, ground_truth: _gt})

                            timgs = decode_pose(_img, tnet_output, threshold=0.0)
                            for i in range(batch_size):
                                io.imsave('../demo_img/epoch{}_step{}_i_{}.jpg'.format(epoch, step, i), timgs[i])
                            print('Epoch {:>2}/{}, step = {:>6}/{:>6}, loss = {:.6f}, time = {}'
                                  .format(epoch, num_epochs, step, int(image_numbers / batch_size), tloss,
                                          time.time() - epoch_time))
                            print('---------------------------------------------------------------------------------')
                    if epoch % save_epoch_number == 0:
                        saver.save(sess, model_folder + 'epoch{}.ckpt'.format(epoch), global_step=global_step)
                        print('Model saved in: {}'.format(model_folder + 'epoch{}.ckpt'.format(epoch)))
            except tf.errors.OutOfRangeError:
                print('End training...')
            finally:
                total_time = time.time() - start_time
                saver.save(sess, modelfile, global_step=global_step)
                print('Model saved as: {}, runing time: {} s'.format(modelfile, total_time))
                print('Done!')

            """
            imgs, kp_vers, kp_hors, kp_classses = sess.run([output, keypoints_ver, keypoints_hor, keypoints_class])
            img = imgs[0]
            kp_ver = kp_vers[0]
            kp_hor = kp_hors[0]
            kp_classs = kp_classses[0]

            dataset.draw_points_on_img(img, point_ver=kp_ver, point_hor=kp_hor, point_class=kp_classs)
            """


if __name__ == '__main__':
    main(gpu_divice='/gpu:0')
