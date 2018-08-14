# -*-coding:utf-8-*-
from ctc.data_gen import decode_sparse_tensor
import tensorflow as tf

# image shape
OUTPUT_SHAPE = (32, 256)

num_epochs = 10000
# LSTM params
num_hidden = 64
num_layer = 1

# 初始化学习率
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 5000
REPORT_STEPS = 100
LEARNING_RATE_DECAY_FACTOR = 0.9
MOMENTUM = 0.9


class img2text(object):
    def __init__(self, sess=None, num_classes=10):
        self.sess = sess if sess is not None else tf.Session()

        # shape[batch,width,height],width用作max_time_step
        self.inputs = tf.placeholder(tf.float32, [None, None, OUTPUT_SHAPE[0]], name='images')
        self.targets = tf.sparse_placeholder(dtype=tf.int32)
        self.seq_len = tf.placeholder(tf.int32, [None])

        # LSTM
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
        # stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layer, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, inputs=self.inputs,
                                       sequence_length=self.seq_len, dtype=tf.float32)
        shape = tf.shape(self.inputs)
        batch_size, max_time_step = shape[0], shape[1]

        # [batch_size*max_time_step,num_hidden]
        outputs = tf.reshape(outputs, [-1, num_hidden])
        logits = tf.layers.dense(outputs, units=num_classes, use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer)
        logits = tf.reshape(logits, [batch_size, -1, num_classes])
        # [max_time_step,batch_size,num_class]
        self.logits = tf.transpose(logits, [1, 0, 2])

        self.loss = self.build_loss()
        # self.cost = tf.reduce_mean(self.loss)

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                                        global_step=self.global_step,
                                                        decay_steps=DECAY_STEPS,
                                                        decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                                        staircase=True)
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate). \
            minimize(self.loss, global_step=self.global_step)

        self.decoded, self.log_prob = self.build_search(is_greedy=True)

        # self.init = tf.global_variables_initializer()
        self.acc_rate = self.build_accurate_rate()

        summarys = []

        summary_hist = [tf.summary.histogram(name=var.name, values=var.value())
                        for var in tf.trainable_variables()]
        summary_loss = tf.summary.scalar(name='train loss', tensor=self.loss)
        summary_acc = tf.summary.scalar(name='accurate', tensor=self.acc_rate)
        summary_image = tf.summary.image(name='input', max_outputs=4,
                                         tensor=tf.expand_dims(self.inputs, axis=-1))
        summarys.append(summary_hist)
        summarys.append(summary_loss)
        summarys.append(summary_acc)
        summarys.append(summary_image)
        self.summary = tf.summary.merge(summarys, name='summary')

        self.saver = tf.train.Saver(max_to_keep=1)

    def restore_model(self, checkpoint_path):
        try:
            ckpt = tf.train.get_checkpoint_state(checkpoint_path)
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('restored done')
        except Exception:
            print("can't restore success")
            self.sess.run(tf.global_variables_initializer())

    def save_model(self, checkpoint_path):
        self.saver.save(self.sess, checkpoint_path, self.global_step)
        print('save done!')

    def build_search(self, is_greedy=True):
        if is_greedy:
            return tf.nn.ctc_greedy_decoder(self.logits, self.seq_len, merge_repeated=True)
        else:
            return tf.nn.ctc_beam_search_decoder(self.logits, self.seq_len, merge_repeated=False)

    def build_loss(self):
        model_loss = tf.nn.ctc_loss(labels=self.targets, inputs=self.logits,
                                    sequence_length=self.seq_len, time_major=True,
                                    preprocess_collapse_repeated=False,
                                    ignore_longer_outputs_than_inputs=True)
        model_loss = tf.reduce_mean(model_loss)
        return model_loss

    def build_accurate_rate(self):
        return tf.reduce_mean(tf.edit_distance(
            tf.cast(self.decoded[0], tf.int32), self.targets))

    def predict(self, images, labels, lengths, digits):
        decode, log_prob, acc = self.sess.run([self.decoded[0], self.log_prob,
                                               self.acc_rate],
                                              feed_dict={
                                                  self.inputs: images,
                                                  self.targets: labels,
                                                  self.seq_len: lengths
                                              })
        report_accuracy(decode, labels, digits)

    def train(self, images, labels, seq_length):
        loss, _, step, summary = self.sess.run([self.loss, self.opt, self.global_step, self.summary],
                                               feed_dict={self.inputs: images,
                                               self.targets: labels,
                                               self.seq_len: seq_length})
        return step, summary, loss

    def test(self, images, labels, seq_length):
        step, summary = self.sess.run([self.global_step, self.summary],
                                      feed_dict={self.inputs: images,
                                                 self.targets: labels,
                                                 self.seq_len: seq_length})
        return step, summary

    def test_run(self, tensor, feed_dict):
        result = self.sess.run(tensor, feed_dict=feed_dict)
        return result


def report_accuracy(decode_list, test_target, digits):
    original_list = decode_sparse_tensor(test_target, digits)
    detected_list = decode_sparse_tensor(decode_list, digits)

    true_numer = 0
    assert len(original_list) == len(detected_list)
    print("T/F: original(length) -- detected(length)")
    for idx, number in enumerate(original_list):
        original_number = ''.join(number)
        detect_number = ''.join(detected_list[idx])
        hit = (original_number == detect_number)
        print(hit, original_number, "(", len(original_number), ") -- ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))
