#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import data_generator

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.python import debug as tf_debug
import deeplab_model
import preprocessing
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model/new',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=26,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=4,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=40000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, default='./datasets/',
                    help='Path to the directory containing the PASCAL VOC data tf record.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_50',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--pre_trained_model', type=str, default='./ini_checkpoints/resnet_v2_50.ckpt',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Whether to use debugger to track down bad values during training.')


_NUM_CLASSES = 2
_HEIGHT = 513
_WIDTH = 513
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

_NUM_IMAGES = {
    'train': 2809,
    'validation': 401,
}

def input_fn(is_training, data_dir, batch_size, should_repeat=False):

    dataset = data_generator.Dataset(is_training=is_training, data_dir=data_dir,
                                     batch_size=batch_size, should_repeat=should_repeat)

    iter = dataset.get_one_shot_iterator()

    image, label = iter.get_next()

    return image, label


def main(unused_argv):


    # Using the Winograd non-fused algorithms provides a small performance boost.
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    if FLAGS.clean_model_dir:
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

    # Set up a RunConfig to only save checkpoints once per training cycle.
    # save checkpoint in every 3 minute
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9).replace(save_checkpoints_secs=300)

    model = tf.estimator.Estimator(
        model_fn=deeplab_model.deeplabv3_plus_model_fn,
        model_dir=FLAGS.model_dir,
        config=run_config,
        params={
            'output_stride': FLAGS.output_stride,
            'batch_size': FLAGS.batch_size,
            'base_architecture': FLAGS.base_architecture,
            'pre_trained_model': FLAGS.pre_trained_model,
            'batch_norm_decay': _BATCH_NORM_DECAY,
            'num_classes': _NUM_CLASSES,
            'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
            'weight_decay': FLAGS.weight_decay,
            'learning_rate_policy': FLAGS.learning_rate_policy,
            'num_train': _NUM_IMAGES['train'],
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': _POWER,
            'momentum': _MOMENTUM,
            'freeze_batch_norm': FLAGS.freeze_batch_norm,
            'initial_global_step': FLAGS.initial_global_step
        })

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensors_to_log = {
            'learning_rate': 'learning_rate',
            'cross_entropy': 'cross_entropy',
            'train_px_accuracy': 'train_px_accuracy',
            'train_mean_iou': 'train_mean_iou',
        }

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=10)
        train_hooks = [logging_hook]
        eval_hooks = None

        if FLAGS.debug:
            debug_hook = tf_debug.LocalCLIDebugHook()
            train_hooks.append(debug_hook)
            eval_hooks = [debug_hook]

        tf.logging.info("Start training.")
        model.train(
            input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, should_repeat=True),
            hooks=train_hooks,
            max_steps=FLAGS.max_iter,
            # steps=1  # For debug
        )

        tf.logging.info("Start evaluation.")
        # Evaluate the model and print results
        eval_results = model.evaluate(
            # Batch size must be 1 for testing because the images' size differs
            input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
            hooks=eval_hooks,
            # steps=1  # For debug
        )
        print(eval_results)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
