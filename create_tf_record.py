"""Converts PASCAL dataset to TFRecords file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import os
import sys

import PIL.Image
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='G:\\imageDatasets\\VOCROOT\\VOC2012',
                    help='Path to the directory containing the PASCAL VOC data.')

parser.add_argument('--output_path', type=str, default='./dataset',
                    help='Path to the directory to create TFRecords outputs.')

parser.add_argument('--train_data_dir', type=str, default='G:\\imageDatasets\\Supervisely Person Dataset\\outputs\\train\\',
                    help='Path to the file listing the training data.')

parser.add_argument('--valid_data_dir', type=str, default='G:\\imageDatasets\\Supervisely Person Dataset\\outputs\\val\\',
                    help='Path to the file listing the validation data.')

parser.add_argument('--image_data_dir', type=str, default='img',
                    help='The directory containing the image data.')

parser.add_argument('--label_data_dir', type=str, default='ann',
                    help='The directory containing the augmented label data.')


def dict_to_tf_example(image_path,
                       label_path):
    """Convert image and label to tf.Example proto.

    Args:
      image_path: Path to a single PASCAL image.
      label_path: Path to its corresponding label.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by image_path is not a valid JPEG or
                  if the label pointed to by label_path is not a valid PNG or
                  if the size of image does not match with that of label.
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG' and image.format != 'PNG':
        raise ValueError('Image format not JPEG or PNG')

    with tf.gfile.GFile(label_path, 'rb') as fid:
        encoded_label = fid.read()
    encoded_label_io = io.BytesIO(encoded_label)
    label = PIL.Image.open(encoded_label_io)
    if label.format != 'PNG':
        raise ValueError('Label format not PNG')

    if image.size != label.size:
        raise ValueError('The size of image does not match with that of label.')

    width, height = image.size

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['png'.encode('utf8')])),
        'label/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_label])),
        'label/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['png'.encode('utf8')])),
    }))
    return example



def create_tf_record(output_filename,
                     image_dir,
                     label_dir,
                     examples):
    """Creates a TFRecord file from examples.

    Args:
      output_filename: Path to where output file is saved.
      image_dir: Directory where image files are stored.
      label_dir: Directory where label files are stored.
      examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 500 == 0:
            tf.logging.info('On image %d of %d', idx, len(examples))
        image_path = os.path.join(image_dir, example)
        label_path = os.path.join(label_dir, example)

        if not os.path.exists(image_path):
            tf.logging.warning('Could not find %s, ignoring example.', image_path)
            continue
        elif not os.path.exists(label_path):
            tf.logging.warning('Could not find %s, ignoring example.', label_path)
            continue

        try:
            tf_example = dict_to_tf_example(image_path, label_path)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            tf.logging.warning('Invalid example: %s, ignoring.', example)

    writer.close()

def list_dir(dir_img: str, dir_ann: str):
    ret = []
    for r, d, f in os.walk(dir_img):
        for file in f:
            if ".png" in file:
                if os.path.exists(os.path.join(dir_ann, file)):
                    ret.append(file)

    return ret



def main(unused_argv):

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    train_examples_img = os.path.join(FLAGS.train_data_dir, FLAGS.image_data_dir)
    train_examples_ann = os.path.join(FLAGS.train_data_dir, FLAGS.label_data_dir)

    val_examples_img = os.path.join(FLAGS.valid_data_dir, FLAGS.image_data_dir)
    val_examples_ann = os.path.join(FLAGS.valid_data_dir, FLAGS.label_data_dir)

    train_examples = list_dir(train_examples_img,
                              train_examples_ann)
    val_examples = list_dir(val_examples_img,
                            val_examples_ann)

    train_output_path = os.path.join(FLAGS.output_path, 'voc_train.record')
    val_output_path = os.path.join(FLAGS.output_path, 'voc_val.record')

    create_tf_record(train_output_path, train_examples_img, train_examples_ann, train_examples)
    create_tf_record(val_output_path, val_examples_img, val_examples_ann, val_examples)



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


