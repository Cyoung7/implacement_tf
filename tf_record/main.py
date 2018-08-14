# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np


def generate_data():
    num = 25
    label = np.asarray(range(0, num))
    images = np.random.random([num, 5, 5, 3])
    print('label size :{}, image size {}'.format(label.shape, images.shape))
    return label, images


def get_batch_data():
    label, images = generate_data()
    images = tf.cast(images, tf.float32)
    label = tf.cast(label, tf.int32)
    input_queue = tf.train.slice_input_producer([images, label], shuffle=False)
    image_batch, label_batch = tf.train.batch(input_queue, batch_size=10, num_threads=1, capacity=64)
    return image_batch, label_batch


image_batch, label_batch = get_batch_data()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)
    i = 0
    try:
        while not coord.should_stop():
            image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
            i += 1
            for j in range(10):
                print(image_batch_v.shape, label_batch_v[j])
    except tf.errors.OutOfRangeError:
        print("done")
    finally:
        coord.request_stop()
    coord.join(threads)
