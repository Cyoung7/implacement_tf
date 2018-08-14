# -*-coding:utf-8-*-
import numpy as np
import freetype
import copy
import random
import cv2

font_path = '/media/cyoung/000E88CC0009670E/projectCV/chinese_image_text/' \
            'img/Img2TextSequence/data_gen/fonts/Deng.ttf'


class put_chinese_text(object):
    def __init__(self, ttf_path):
        self._face = freetype.Face(ttf_path)

    def draw_text(self, image, pos, text, text_size, text_color):
        """
        draw chinese text with ttf
        :param image: ndarray
        :param pos: where is draw text
        :param text: the context,for chinese be unicode type
        :param text_size: text size
        :param text_color: text color
        :return: image
        """
        self._face.set_char_size(text_size * 64)
        metrics = self._face.size
        ascender = metrics.ascender / 64.0

        # descender = metrics.descender / 64.0
        # height = metrics.height / 64.0
        # linegap = height - ascender + descender

        y_pos = int(ascender)
        if not isinstance(text, str):
            text = text.decode('utf-8')
        img = self.draw_string(image, pos[0], pos[1] + y_pos, text, text_color)
        return img

    def draw_string(self, img, x_pos, y_pos, text, color):
        """
        draw string
        :param img: ndarray
        :param x_pos: x coordinate
        :param y_pos: y coordinate
        :param text:  text(str)
        :param color: text color
        :return: image
        """
        prev_char = 0
        pen = freetype.Vector()
        pen.x = x_pos << 6
        pen.y = y_pos << 6
        hscale = 1.0
        matrix = freetype.Matrix(int(hscale) * 0x10000, int(0.2 * 0x10000),
                                 int(0.0 * 0x10000), int(1.1 * 0x10000))
        cur_pen = freetype.Vector()
        pen_translate = freetype.Vector()
        image = copy.deepcopy(img)

        for cur_char in text:
            self._face.set_transform(matrix, pen_translate)
            self._face.load_char(cur_char)
            # kerning = self._face.get_kerning(prev_char, cur_char)
            pen.x = pen.x + 170
            slot = self._face.glyph
            bitmap = slot.bitmap

            cur_pen.x = pen.x
            cur_pen.y = pen.y - slot.bitmap_top * 64
            self.draw_ft_bitmap(image, bitmap, cur_pen, color)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)
            pen.x += slot.advance.x
            prev_char = cur_char

        return image

    def draw_ft_bitmap(self, img, bitmap, pen, color):
        """
        draw each char
        :param img: ndarray
        :param bitmap:
        :param pen:
        :param color: pen color e.g.(0,0,255)-->red
        :return: image
        """
        x_pos = pen.x >> 6
        y_pos = pen.y >> 6
        cols = bitmap.width
        rows = bitmap.rows

        glyph_pixels = bitmap.buffer

        for row in range(rows):
            for col in range(cols):
                if glyph_pixels[row * cols + col] != 0:
                    img[y_pos + row][x_pos + col][0] = color[0]
                    img[y_pos + row][x_pos + col][1] = color[1]
                    img[y_pos + row][x_pos + col][2] = color[2]
        # return img


class gen_id_card(object):
    def __init__(self, ttf_path):
        self.number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.char_set = self.number
        self.len = len(self.char_set)

        self.max_size = 18
        self.ft = put_chinese_text(ttf_path)

    def gen_text(self, is_random=False):
        text = ''
        vecs = np.zeros((self.max_size * self.len))

        # 生成不定长的字符串
        if is_random:
            size = random.randint(8, self.max_size)
        else:
            size = self.max_size
        for i in range(size):
            c = random.choice(self.char_set)
            vec = self.char2vec(c)
            text = text + c
            vecs[i * self.len:(i + 1) * self.len] = np.copy(vec)
        return text, vecs

    def char2vec(self, c):
        vec = np.zeros(self.len)
        for j in range(self.len):
            if self.char_set[j] == c:
                vec[j] = 1
        return vec

    def vec2text(self, vecs):
        text = ''
        v_len = len(vecs)
        for i in range(v_len):
            if vecs[i] == 1:
                text = text + self.number[i % self.len]
        return text

    def gen_one_image(self, is_random=False):
        text, vec = self.gen_text(is_random)
        img = np.zeros((32, 256, 3))
        color_ = (255, 255, 255)

        pos = (0, 0)
        text_size = 21
        image = self.ft.draw_text(img, pos, text, text_size, color_)
        return image[:, :, 2], text, vec

    def get_next_batch(self, batch_size=128, output_size=(32, 256)):
        inputs = np.zeros((batch_size, output_size[1], output_size[0]))
        codes = []
        for i in range(batch_size):
            image, text, vec = self.gen_one_image(True)
            inputs[i, :, :] = np.transpose(image)
            codes.append(list(text))

        targets = [np.asarray(i) for i in codes]
        sparse_targets = sparse_tuple_from(targets)
        seq_len = np.ones(inputs.shape[0]) * output_size[1]
        # [batch_size,256,32] seq_len:[batch_size,]:[256,256,...]
        return inputs, sparse_targets, seq_len

    def gen_anymore_image(self):
        while True:
            yield self.gen_one_image()


# 作用于sparse_tensor
def sparse_tuple_from(sequences, dtype=np.int32):
    indices = []
    values = []
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), list(range(len(seq)))))
        values.extend(seq)

    # shape:(n,2)
    indices = np.asarray(indices, dtype=np.int64)
    # shape:(n,)
    values = np.asarray(values, dtype=dtype)
    # shape:(2,)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1])
    return indices, values, shape


def decode_sparse_tensor(sparse_tensor, digits):
    """

    :param sparse_tensor: tuple(indices, values, shape)
    :param label_list:
    :return:
    """

    decoded_indexes = list()

    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor, digits))
    return result


def decode_a_seq(indexes, spars_tensor,digits):
    decoded = []
    for m in indexes:
        char = digits[spars_tensor[1][m]]
        decoded.append(char)
    return decoded



if __name__ == '__main__':

    gen_obj = gen_id_card(font_path)

    for image, label, vec in gen_obj.gen_anymore_image():
        cv2.imshow('image', image)
        cv2.waitKey()
