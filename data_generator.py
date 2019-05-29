#!/usr/bin/env python
# -*- coding: utf-8 -*-

import collections
import os
import tensorflow as tf
import preprocessing
import io
import PIL

class Dataset(object):

    def __init__(self, is_training, data_dir, batch_size, should_shuffle=False):
        self._MIN_SCALE = 0.5
        self._MAX_SCALE = 2.0
        self._HEIGHT = 513
        self._WIDTH = 513
        self._IGNORE_LABEL = 255
        self.is_training = is_training
        self.data_dir = data_dir
        self.num_readers = 2
        self.should_repeat = False
        self.should_shuffle = should_shuffle
        self.batch_size = batch_size

    def _parse_record(self, raw_record):
        """Parse PASCAL image and label from a tf record."""
        keys_to_features = {
            'image/height':
                tf.FixedLenFeature((), tf.int64),
            'image/width':
                tf.FixedLenFeature((), tf.int64),
            'image/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
                tf.FixedLenFeature((), tf.string, default_value='png'),
            'label/encoded':
                tf.FixedLenFeature((), tf.string, default_value=''),
            'label/format':
                tf.FixedLenFeature((), tf.string, default_value='png'),
        }

        parsed = tf.parse_single_example(raw_record, keys_to_features)

        # height = tf.cast(parsed['image/height'], tf.int32)
        # width = tf.cast(parsed['image/width'], tf.int32)

        image = tf.image.decode_image(
            tf.reshape(parsed['image/encoded'], shape=[]), 3)
        image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
        image.set_shape([None, None, 3])

        label = tf.image.decode_image(
            tf.reshape(parsed['label/encoded'], shape=[]), 1)
        label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
        label.set_shape([None, None, 1])

        return image, label

    def _preprocess_image(self, image, label, is_training):
        """Preprocess a single image of layout [height, width, depth]."""
        if is_training:
            # Randomly scale the image and label.
            image, label = preprocessing.random_rescale_image_and_label(
                image, label, self._MIN_SCALE, self._MAX_SCALE)

            # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
            image, label = preprocessing.random_crop_or_pad_image_and_label(
                image, label, self._HEIGHT, self._WIDTH, self._IGNORE_LABEL)

            # Randomly flip the image and label horizontally.
            image, label = preprocessing.random_flip_left_right_image_and_label(
                image, label)

            image.set_shape([self._HEIGHT, self._WIDTH, 3])
            label.set_shape([self._HEIGHT, self._WIDTH, 1])

        image = preprocessing.mean_image_subtraction(image)

        return image, label

    def get_filenames(self):
        """Return a list of filenames.

        Args:
          is_training: A boolean denoting whether the input is for training.
          data_dir: path to the the directory containing the input data.

        Returns:
          A list of file names.
        """
        if self.is_training:
            return [os.path.join(self.data_dir, 'voc_train.record')]
        else:
            return [os.path.join(self.data_dir, 'voc_val.record')]

    def get_one_shot_iterator(self):
        """Gets an iterator that iterates across the dataset once.
        Returns:
          An iterator of type tf.data.Iterator.
        """

        files = self.get_filenames()

        dataset = (
            tf.data.TFRecordDataset(files, num_parallel_reads=self.num_readers)
                .map(self._parse_record, num_parallel_calls=self.num_readers)
                .map(lambda image, label: self._preprocess_image(image, label, self.is_training),
                     num_parallel_calls=self.num_readers))

        if self.should_shuffle:
            dataset = dataset.shuffle(buffer_size=100)

        if self.should_repeat:
            dataset = dataset.repeat()  # Repeat forever for training.
        else:
            dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)

        return dataset.make_one_shot_iterator()




