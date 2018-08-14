# -*-coding:utf-8-*-
from ctc.data_gen import gen_id_card, font_path
from ctc.model import img2text
import tensorflow as tf

BATCHES = 10
BATCH_SIZE = 64
TRAIN_SIZE = BATCH_SIZE * BATCHES

if __name__ == '__main__':
    obj = gen_id_card(font_path)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session()

    i2t = img2text(sess, num_classes=11)
    i2t.restore_model('./output/model')
    fw_train = tf.summary.FileWriter('./output/log/train', flush_secs=20)
    fw_test = tf.summary.FileWriter('./output/log/test', flush_secs=20)
    fw_train.add_graph(sess.graph)
    sess.graph.finalize()
    step = 1

    while True:
        images, sparse_targets, seq_len = obj.get_next_batch(BATCH_SIZE)

        # feed_dict = {i2t.inputs: images, i2t.targets: sparse_targets, i2t.seq_len: seq_len}
        # tensor_list = [i2t.loss]
        # result = i2t.test_run(tensor_list, feed_dict)

        if step % 10 == 0:
            step, summary = i2t.test(images, sparse_targets, seq_len)
            fw_test.add_summary(summary, step)

            i2t.predict(images, sparse_targets, seq_len, obj.number)

            if step % 10 == 0:
                i2t.save_model('./output/model/ckpt')
        step, summary, loss = i2t.train(images, sparse_targets, seq_len)
        fw_train.add_summary(summary, step)
        print('step:', step, ' ', 'train loss:', loss)
        print('test git')
