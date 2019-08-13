"""
This file is used to generate TFRecords using the AI Challenger dataset.

@ Author: Yu Sun. vxallset@outlook.com

@ Date created: Jun 04, 2019

@ Last modified: Jun 27, 2019

"""
import numpy as np
import os
import time
from skimage import io, draw
from skimage.transform import resize
from random import shuffle
import tensorflow as tf
import json

"""
# The image which contains a person is collected from the AI Challenger dataset in the following steps:
    1. Get the coordinate of the bounding box in the original image.
    2. Adjust the ratio of the bounding box to be 4:3 (height : width)

    Note that the coordinates of keypoints are also re-calculated when the foreground parts are clipped from the 
    original images.

"""


def draw_points_on_img(img, point_ver, point_hor, point_class):
    for i in range(len(point_class)):
        if point_class[i] != 3:
            rr, cc = draw.circle(point_ver[i], point_hor[i], 10, (256, 192))
            #draw.set_color(img, [rr, cc], [0., 0., 0.], alpha=5)
            img[rr, cc, :] = 0
    #io.imshow(img)
    #io.show()

    return img


def draw_lines_on_img(img, point_ver, point_hor, point_class):
    line_list = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10],
               [10, 11], [12, 13], [13, 6], [13, 9], [13, 0], [13, 3]]

    # key point class: 1:visible, 2: not visible, 3: not marked
    for start_point_id in range(len(point_class)):
        if point_class[start_point_id] == 3:
            continue
        for end_point_id in range(len(point_class)):
            if point_class[end_point_id] == 3:
                continue

            if [start_point_id, end_point_id] in line_list:
                rr, cc = draw.line(int(point_ver[start_point_id]), int(point_hor[start_point_id]),
                                   int(point_ver[end_point_id]), int(point_hor[end_point_id]))
                draw.set_color(img, [rr, cc], [255, 0, 0])

    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def extract_people_from_dataset(dataset_root_path='../../../dataset/ai_challenger/', image_save_path='../dataset/imgs/',
                                tfrecords_path='../dataset/', is_shuffle=True):
    """
    This function is used to extract people from the AI Challenger dataset. The extract image will contain only one
    person each and will be saved as a single .jpg file. At last, the image and the the corresponding annotation will
    be saved into a .tfrecord file.

    :param dataset_root_path: the root path of the AI Challenger dataset.
    :param image_save_path: the path used to save the clipped images.
    :param tfrecord_path: the path used to save the .tfrecords file.
    :param is_shuffle: is shuffle.
    :return: None.
    """
    annotation_file = os.path.join(dataset_root_path, 'keypoint_train_annotations_20170909.json')
    image_read_path = os.path.join(dataset_root_path, 'train_images')
    tfrecords_file = os.path.join(tfrecords_path, 'train.tfrecords')

    if not os.path.exists(tfrecords_path):
        os.mkdir(tfrecords_path)
    if os.path.exists(tfrecords_file):
        os.remove(tfrecords_file)
    if os.path.exists(image_save_path):
        useless = os.listdir(image_save_path)
        for onefile in useless:
            os.remove(os.path.join(image_save_path, onefile))
    else:
        os.mkdir(image_save_path)

    saved_number = 0
    image_number = 0
    start_time = time.time()
    with tf.python_io.TFRecordWriter(tfrecords_file) as tfwriter:

        with open(annotation_file, 'r') as jsfile:
            data = json.load(jsfile)

            for one_item in data:
                img_id = one_item['image_id']
                image_number += 1
                if image_number % 100 == 0:
                    print('Processed {} images, extracted {} people from the dataset. '
                          'time = {}'.format(image_number, saved_number, time.time() - start_time))

                kps = one_item['keypoint_annotations']
                boxes = one_item['human_annotations']

                # read image
                img_filename = os.path.join(image_read_path, img_id + '.jpg')
                img = io.imread(img_filename)

                for i in range(len(boxes)):
                    # construct the name of a human in the dictionary,
                    # for example, the first one (when i = 0) is 'human1'
                    human_name = 'human' + str(i+1)

                    kp = kps[human_name]
                    box = boxes[human_name]
                    p1_hor, p1_ver, p2_hor, p2_ver = box
                    foreground = img[p1_ver:p2_ver, p1_hor:p2_hor, :]

                    try:
                        foreground = resize(foreground, (256, 192, 3))
                    except ValueError:
                        print('ValueError at image {} and {}'.format(image_number, human_name))
                        continue

                    foreground = foreground * 255.0
                    foreground_uint8 = np.uint8(foreground)

                    kp_hor = (np.array(kp[0::3]) - p1_hor) / (p2_hor - p1_hor) * 192
                    kp_ver = (np.array(kp[1::3]) - p1_ver) / (p2_ver - p1_ver) * 256
                    kp_class = np.array(kp[2::3])

                    img_name = img_id + '_' + human_name + '.jpg'

                    io.imsave(os.path.join(image_save_path, img_id + '_' + human_name + '.jpg'), foreground_uint8)

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'image_name': _bytes_feature(img_name.encode()),
                                'image_raw': _bytes_feature(foreground_uint8.tobytes()),
                                'keypoints_ver': _bytes_feature(np.uint8(kp_ver).tobytes()),
                                'keypoints_hor': _bytes_feature(np.uint8(kp_hor).tobytes()),
                                'keypoints_class': _bytes_feature(np.uint8(kp_class).tobytes())
                            }))
                    tfwriter.write(example.SerializeToString())

                    saved_number += 1
    print('Extracted {} people from the dataset in total.'.format(saved_number))


def decode_proto(proto):
    features = tf.parse_single_example(proto,
                                       features={
                                           'image_name': tf.FixedLenFeature([], tf.string),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'keypoints_ver': tf.FixedLenFeature([], tf.string),
                                           'keypoints_hor': tf.FixedLenFeature([], tf.string),
                                           'keypoints_class': tf.FixedLenFeature([], tf.string),
                                       })
    image_name = features['image_name']

    image_raw = tf.decode_raw(features['image_raw'], out_type=np.uint8)
    image = tf.reshape(image_raw, [256, 192, 3])

    keypoints_ver = tf.decode_raw(features['keypoints_ver'], out_type=np.uint8)
    keypoints_hor = tf.decode_raw(features['keypoints_hor'], out_type=np.uint8)
    keypoints_class = tf.decode_raw(features['keypoints_class'], out_type=np.uint8)
    return image_name, image, keypoints_ver, keypoints_hor, keypoints_class


def decode_tfrecord(filename_queue):
    tfreader = tf.TFRecordReader()
    _, proto = tfreader.read(filename_queue)
    image_name, image, keypoints_ver, keypoints_hor, keypoints_class = decode_proto(proto)

    return image_name, image, keypoints_ver, keypoints_hor, keypoints_class


def input_batch(datasetname, batch_size, num_epochs):
    """
    This function is used to decode the TFrecord and return a batch of images as well as their information
    :param datasetname: the name of the TFrecord file.
    :param batch_size: the number of images in a batch
    :param num_epochs: the number of epochs
    :return: a batch of images as well as their information
    """
    with tf.name_scope('input_batch'):
        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        mydataset = tf.data.TFRecordDataset(datasetname)
        mydataset = mydataset.map(decode_proto)

        # have no idea why I can't set the parameter of mydataset.shuffle to be the number of the dataset......
        # mydataset = mydataset.shuffle(200)
        mydataset = mydataset.repeat(num_epochs * 2)
        # drop all the data that can't be used to make up a batch
        mydataset = mydataset.batch(batch_size, drop_remainder=True)
        iterator = mydataset.make_one_shot_iterator()

        nextelement = iterator.get_next()
        return nextelement


def mytest():
    tfrecord_file = '../dataset/train.tfrecords'

    filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=None)
    image_name, image, keypoints_ver, keypoints_hor, keypoints_class = decode_tfrecord(filename_queue)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            # while not coord.should_stop():
            for i in range(10):
                img_name, img, point_ver, point_hor, point_class = sess.run([image_name, image, keypoints_ver,
                                                                             keypoints_hor, keypoints_class])

                print(img_name, point_hor, point_ver, point_class)

                for i in range(len(point_class)):
                    if point_class[i] > 0:
                        rr, cc = draw.circle(point_ver[i], point_hor[i], 10, (256, 192))
                        img[rr, cc, :] = 0

                io.imshow(img)
                io.show()

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()


if __name__ == '__main__':
    extract_people_from_dataset()
    #mytest()


