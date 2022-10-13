#!/usr/bin/env python
# coding: utf-8
# author email: erfan.molaei@gmail.com
import os
import pickle
import joblib

import numpy as np
from scipy.stats import norm

import tensorflow_addons as tfa
import tensorflow as tf
import datetime

from common.tf.TFBaseModel import TFBaseModel
from common.tf.layers.DenseLayer import DenseLayer
from common.tf.layers.ReshapeLayer import ReshapeLayer
from common.tf.optimizers.Trainer import Trainer
from tensorflow.compat.v1.losses import Reduction
from utils import load_datasets, encode_sequences
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


# METRIC = "val_loss"

def __create_callbacks__(metric, ld=6, epochs=1, optimizer='adam', batch_size=1):
    # cpk_path = f'./drive/My Drive/ShiLab/training_checkpoints/HLA/mixed_pop/mixed_best_model_{kfold}.h5'

    # checkpoint = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=cpk_path,
    #     monitor= metric,
    #     mode='min',
    #     save_best_only=True,
    #     verbose=1,
    # )

    log_dir = "logs/fit/" + f"latent_dim_{ld}_bs_{batch_size}_epochs_{epochs}_optimizer_{optimizer}_" + \
              datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric,
        mode='min',
        factor=0.2,
        patience=50,
        verbose=0
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        mode='min',
        patience=75,
        verbose=1,
        restore_best_weights=True
    )

    # callbacks = [
    #     checkpoint,
    #     reducelr,
    #     earlystop,
    #     tensorboard_callback]

    callbacks = [
        # checkpoint,
        # earlystop,
        tensorboard_callback]

    return callbacks


@tf.function()
def _data_mapper(X_sample, q):
    # return tf.one_hot(X_sample, q)
    return tf.reshape(tf.one_hot(X_sample, q), [X_sample.shape[0], q, 1])
    # return tf.reshape(tf.one_hot(X_sample, q), [-1])


def _get_dataset(X, bs, q, training=True):
    AUTO = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((X))
    if training:
        dataset = dataset.shuffle(X.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    # Add Attention Mask
    dataset = dataset.map(lambda x: _data_mapper(x, q), num_parallel_calls=AUTO, deterministic=False)
    # Prefetech to not map the whole dataset
    dataset = dataset.prefetch(AUTO)
    dataset = dataset.batch(bs, drop_remainder=False)
    return dataset


class BaseVAE(TFBaseModel):
    def __init__(self, params):
        # super(BaseVAE, self).__init__(**kwargs)
        super(BaseVAE, self).__init__(
            mixed_precision=params["model"]["mixed_precision"]
        )

        self.optimizers = {'adam': tf.keras.optimizers.Adam(),
                           'rmsprop': tf.keras.optimizers.RMSprop(),
                           'sgd': tf.keras.optimizers.SGD()}

        ### Model params
        mparams = params["model"]
        self.optimizer = mparams["optimizer"]
        self.METRIC = mparams["metric"]
        self.N = mparams["n"]
        self.activation = mparams["activation"]
        self.save_root_dir = params["runconfig"]["model_dir"] + os.path.sep

        self.L, self.q = mparams["l"], mparams["q"]
        self.batch_size, self.latent_dim = mparams["batch_size"], mparams["latent_dim"]
        # self.z_mean = None
        # self.z_log_var = None
        # self._sampling = None
        # self.vae = None
        # self.hist = None

        # CS util params for layers
        self.tf_summary = mparams["tf_summary"]

        self.mixed_precision = mparams["mixed_precision"]
        # Model trainer
        self.trainer = Trainer(
            params=params["optimizer"],
            tf_summary=self.tf_summary,
            mixed_precision=self.mixed_precision,
        )

    def _enc_dec_layers(self, **kwargs):
        raise NotImplementedError()

    def __sampling(self, z_mean, z_log_var):
        with tf.compat.v1.name_scope(f"sampleing_layer"):
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            result = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        return result

    def build_model(self, features, mode):
        x = features
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        enc_layers, dec_layers = self._enc_dec_layers()

        # Build the encoder
        # encoder_inputs = x = tf.keras.layers.Input(shape=(self.L, self.q, 1))
        with tf.compat.v1.name_scope("encoder"):
            for layer in enc_layers:
                x = layer(x)

        with tf.compat.v1.name_scope("encoder_out"):
            z_mean = DenseLayer(self.latent_dim, name="z_mean")(x)
            z_log_var = DenseLayer(self.latent_dim, name="z_log_var")(x)
            _sampling = self.__sampling(z_mean, z_log_var)
        x = _sampling
        # Build the decoder
        with tf.compat.v1.name_scope("decoder"):
            for layer in dec_layers:
                x = layer(x)
        decoder_outputs = ReshapeLayer((self.L, self.q, 1))(x)
        return z_mean, z_log_var, _sampling, decoder_outputs

    def build_total_loss(self, model_outputs, features, labels, mode):
        z_mean, z_log_var, z, decoder_outputs = model_outputs
        # Flatten the logits
        decoder_outputs_casted = tf.compat.v1.cast(
            decoder_outputs, dtype="float16" if self.mixed_precision else "float32"
        )
        features_casted = tf.compat.v1.cast(
            features, dtype="float16" if self.mixed_precision else "float32"
        )
        is_training = mode == tf.estimator.ModeKeys.TRAIN

        reconstruction_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            features_casted,
            decoder_outputs_casted,
            loss_collection=None,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        return total_loss

    def build_train_ops(self, total_loss):
        """
        Setup optimizer and build train ops.
        """
        return self.trainer.build_train_ops(total_loss)

    def build_eval_metric_ops(self, model_outputs, labels, features):
        """
        Evaluation metrics
        """
        z_mean, z_log_var, z, decoder_outputs = model_outputs
        # Flatten the logits
        decoder_outputs_casted = tf.compat.v1.cast(
            decoder_outputs, dtype="float16" if self.mixed_precision else "float32"
        )
        features_casted = tf.compat.v1.cast(
            features, dtype="float16" if self.mixed_precision else "float32"
        )

        reconstruction_loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            features_casted,
            decoder_outputs_casted,
            loss_collection=None,
            reduction=Reduction.SUM_OVER_BATCH_SIZE,
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        hamming_loss = tfa.metrics.hamming_loss_fn(features_casted, decoder_outputs_casted, threshold=0.6, mode='multiclass')
        total_loss = reconstruction_loss + kl_loss

        metrics_dict = dict()

        metrics_dict["eval/kl_loss"] = kl_loss
        metrics_dict["eval/reconstruction_loss"] = reconstruction_loss
        metrics_dict["eval/hamming_loss"] = hamming_loss
        metrics_dict["eval/total_loss"] = total_loss
        return metrics_dict

    def get_evaluation_hooks(self, logits, labels, features):
        """ As a result of this TF issue, need to explicitly define summary
        hooks to able to log image summaries in eval mode
        https://github.com/tensorflow/tensorflow/issues/15332
        """
        if self.log_image_summaries:
            input_image = features
            reshaped_mask_image = labels
            reshaped_mask_image = tf.cast(reshaped_mask_image, dtype=tf.int32)
            self._write_image_summaries(
                logits, input_image, reshaped_mask_image, is_training=False,
            )
            summary_hook = tf.estimator.SummarySaverHook(
                save_steps=1,
                output_dir=self.output_dir,
                summary_op=tf.compat.v1.summary.merge_all(),
            )
            return [summary_hook]
        else:
            return None
