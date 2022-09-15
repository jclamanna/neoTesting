#!/usr/bin/env python
# coding: utf-8
# author email: erfan.molaei@gmail.com
import os
import pickle
import joblib

import numpy as np
from scipy.stats import norm

import tensorflow as tf
from tensorflow.keras import layers
import datetime

from utils import load_datasets, encode_sequences
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt 

# from Loss import plotLoss
# from Latent import plotLatent1d, plotLatent2d

# class metricsByEpoch(keras.callbacks.Callback):

#     def on_epoch_end(self, epoch, target_epochs):
#         if epoch in target_epoch:

       

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
        patience=10,
        verbose=0
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor=metric,
        mode='min',
        patience=25,
        verbose=1,
        restore_best_weights=True
    )

    callbacks = [
        #  checkpoint,
        #         reducelr,
        #         earlystop,
        tensorboard_callback]

    return callbacks


@tf.function()
def _data_mapper(X_sample, q):
    # return tf.one_hot(X_sample, q)
    return tf.reshape(tf.one_hot(X_sample, q), [X_sample.shape[0], q, 1])
    # return tf.reshape(tf.one_hot(X_sample, q), [-1])


def _get_dataset(X, bs, q, training=True):
    AUTO = tf.data.experimental.AUTOTUNE
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

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, training=None):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseVAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(BaseVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self._set_inputs(inputs=self.encoder.inputs, outputs=decoder.outputs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.hamming_loss_tracker = tf.keras.metrics.Mean(name="hamming_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.hamming_loss_tracker,
        ]

    #def call(self, inputs, training=None):
    #    return self.decoder(self.encoder(inputs, training=training)[-1], training=training)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # hamming_loss = tfa.metrics.hamming_loss_fn(data, reconstruction, threshold=0.6, mode='multiclass')
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # self.hamming_loss_tracker.update_state(hamming_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # "hamming_loss": self.hamming_loss_tracker.result(),
        }

    def test_step(self, data):
        # Compute predictions
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        # Updates the metrics tracking the loss
        # hamming_loss = tfa.metrics.hamming_loss_fn(data, reconstruction, threshold=0.6, mode='multiclass')
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # self.hamming_loss_tracker.update_state(hamming_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            # "hamming_loss": self.hamming_loss_tracker.result(),
        }


class VAE:
    """
        Abstract Base class for all VAEs.  Does not specify layers, you must
        subclass it and provide an _enc_dec_layers method. The inputs are
        shaped as (L, q, 1) and the outputs of decoder layers should be of size L*q in any shape.
    """

    def __init__(self, save_path='saved models', **kwargs):
        self.optimizer = None
        self.METRIC = None
        self.N = None
        self.save_root_dir = save_path + os.path.sep
        self.optimizers = {'adam': tf.keras.optimizers.Adam(),
                           'rmsprop': tf.keras.optimizers.RMSprop(),
                           'sgd': tf.keras.optimizers.SGD()}
        self.L, self.q = None, None
        self.batch_size, self.latent_dim = None, None
        self.z_mean = None
        self.z_log_var = None
        self._sampling = None
        self.vae = None
        self.hist = None

    def instantiate_model(self, L, q, latent_dim, batch_size, activation, **kwargs):
        self.L, self.q = L, q
        self.batch_size, self.latent_dim = batch_size, int(latent_dim)
        enc_layers, dec_layers = self._enc_dec_layers(self.L, self.q,
                                                      self.latent_dim, self.batch_size,
                                                      activation=activation, **kwargs)

        # Build the encoder
        encoder_inputs = x = tf.keras.layers.Input(shape=(self.L, self.q, 1))
        for layer in enc_layers:
            x = layer(x)

        self.z_mean = layers.Dense(self.latent_dim, name="z_mean")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        self._sampling = z = Sampling()([self.z_mean, self.z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [self.z_mean, self.z_log_var, z], name="encoder")

        # Build the decoder
        latent_inputs = x = tf.keras.layers.Input(shape=(self.latent_dim,))
        for layer in dec_layers:
            x = layer(x)
        decoder_outputs = layers.Reshape((self.L, self.q, 1))(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
        self.vae = BaseVAE(encoder, decoder, **kwargs)

    def _enc_dec_layers(self, L, q, latent_dim, batch_size, activation, **kwargs):
        raise NotImplementedError()

    def train_vae(self, train_seqs, epochs, optimizer,
                  val_seqs=None, save_history=True, verbose=2,
                  use_callbacks=True):
        """
            train_seqs: ndarray of shape (n_samples, L)
            val_seqs: ndarray of shape (n_samples, L)
            optimizer: string
        """
        if val_seqs is not None:
            self.METRIC = "val_loss"
        else:
            self.METRIC = "loss"

        assert (self.L == train_seqs.shape[1])
        self.N = train_seqs.shape[0]
        self.optimizer = self.optimizers[optimizer]
        steps_per_epoch = np.ceil(train_seqs.shape[0] / self.batch_size)
        validation_steps = np.ceil(val_seqs.shape[0] / self.batch_size)
        assert (train_seqs.shape[0] <= val_seqs.shape[0])

        x_train = _get_dataset(train_seqs, self.batch_size, self.q, training=True)
        x_valid = _get_dataset(val_seqs, self.batch_size, self.q, training=False)

        callbacks = __create_callbacks__(metric=self.METRIC,
                                         ld=self.latent_dim,
                                         epochs=epochs,
                                         optimizer=self.optimizer._name,
                                         batch_size=self.batch_size)
        # super(BaseVAE, self).build(input_shape)
        self.vae.compile(optimizer=self.optimizer)
        self.hist = self.vae.fit(x_train,
                            epochs=epochs,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=validation_steps,
                            validation_data=x_valid,
                            callbacks=callbacks if use_callbacks else None,
                            verbose=verbose,
                            )
        # if save_history:
        #     with open(
        #             f"logs/logs_Latent_{self.latent_dim}_batch_size_"
        #             f"{self.batch_size}_epochs_{epochs}_optimizer_{self.optimizer._name}" + ".pkl",
        #             'wb') as f:
        #         pickle.dump(hist.history, f)

    def summarize(self):
        self.vae.encoder.summary()
        self.vae.decoder.summary()

    def save_model(self, name, path):
        print("Saving Model...")
        path = path +"/model/"
        tf.keras.models.save_model(self.vae.encoder, path + name + "_enc", save_format="tf", include_optimizer=True)
        
        tf.keras.models.save_model(self.vae.decoder, path + name + "_dec", save_format="tf")

        with open(path + '{}_param.pkl'.format(name), 'wb') as f:
            d = (self.batch_size, self.L, self.q, self.latent_dim,
                 self.vae.optimizer.get_config(), self.__class__.__name__)
            joblib.dump(d, f)
        
        print("Model Saved Successfully!")

    def _extract_layers(self):
        # self.encoder = self.vae.get_layer('encoder')
        # self.decoder = self.vae.get_layer('decoder')
        self.z_mean = self.vae.encoder.get_layer('z_mean').output
        self.z_log_var = self.vae.encoder.get_layer('z_log_var').output
        self._sampling = Sampling()([self.z_mean, self.z_log_var])

    def load_model(self, name, path):
        print("Loading Model...")
        path = path + "/model/"
        with open(path + '{}_param.pkl'.format(name), 'rb') as f:
            d = joblib.load(f)
            self.batch_size, self.L, self.q, self.latent_dim, opt_config, cls = d

        encoder = tf.keras.models.load_model(path + name + "_enc", compile=False)
        decoder = tf.keras.models.load_model(path + name + "_dec", compile=False)
        self.vae = BaseVAE(encoder, decoder)

        self._extract_layers()
        # self.vae.optimizer.from_config(opt_config)
        print("Model Loaded Successfully!")
        pass

    def encode(self, data):
        z_mean, z_log_var, _ = self.vae.encoder.predict(tf.keras.utils.to_categorical(data, self.q))
        return z_mean, z_log_var

    def decode_bernoulli(self, z):
        brnll = self.vae.decoder.predict(z)
        brnll = brnll.reshape((z.shape[0], self.L, self.q))
        # clip like in Keras categorical_crossentropy used in vae_loss
        brnll = np.clip(brnll, 1e-7, 1 - 1e-7)
        brnll = brnll / np.sum(brnll, axis=-1, keepdims=True)
        return brnll

    def single_sample(self, data):
        return self.vae.predict(tf.keras.utils.to_categorical(data, self.q))

    def lELBO(self, seqs, n_samples=1000):
        N, L = seqs.shape
        rN, rL = np.arange(N)[:,None], np.arange(L)

        zm, zlv = self.vae.encode(seqs)
        zstd = np.exp(zlv/2)

        kl_loss = 0.5*np.sum(1 + zlv - np.square(zm) - np.exp(zlv), axis=-1)

        xent_loss = np.zeros(N, dtype=float)
        for n in range(n_samples):
            z = norm.rvs(zm, zstd)
            brnll = self.decode_bernoulli(z)
            xent_loss += np.sum(-np.log(brnll[rN, rL, seqs]), axis=-1)
        xent_loss /= n_samples

        return xent_loss - kl_loss

    def logp(self, seqs, n_samples=1000):
        N, L = seqs.shape
        rN, rL = np.arange(N)[:,None], np.arange(L)

        zm, zlv = self.vae.encode(seqs)
        zstd = np.exp(zlv/2)

        logp = None
        for n in range(n_samples):
            z = norm.rvs(zm, zstd)
            brnll = self.decode_bernoulli(z)

            lqz_x = np.sum(norm.logpdf(z, zm, zstd), axis=-1)
            lpx_z = np.sum(np.log(brnll[rN, rL, seqs]), axis=-1)
            lpz = np.sum(norm.logpdf(z, 0, 1), axis=-1)
            lpxz = lpz + lpx_z

            if logp is None:
                logp = lpxz - lqz_x
            else:
                np.logaddexp(logp, lpxz - lqz_x, out=logp)

        return logp - np.log(n_samples)

    def generate(self, N):
        # returns a generator yielding sequences in batches
        assert(N % self.batch_size == 0)

        print("")
        for n in range(N // self.batch_size):
            print("\rGen {}/{}".format(n*self.batch_size, N), end='')

            z = norm.rvs(0., 1., size=(self.batch_size, self.latent_dim))
            brnll = self.decode_bernoulli(z)

            c = np.cumsum(brnll, axis=2)
            c = c/c[:,:,-1,None] # correct for fp error
            r = np.random.rand(self.batch_size, self.L)

            seqs = np.sum(r[:,:,None] > c, axis=2, dtype='u1')
            yield seqs
        print("\rGen {}/{}   ".format(N, N))

    def getLoss(self):
        loss = {}

        loss['kl'] = self.hist.history['kl_loss']
        loss['val_kl'] = self.hist.history['val_kl_loss']
        loss['rec'] = self.hist.history['reconstruction_loss']
        loss['val_rec'] = self.hist.history['val_reconstruction_loss']
        loss['total'] = self.hist.history['loss']
        loss['val_total'] = self.hist.history['val_loss']

        return loss

    # def get_config(self):
    #     return{"optimizer": self.optimizer,
    #            "METRIC": self.METRIC,
    #            "N": self.N,               
    #            "save_root_dir": self.save_root_dir,
    #            "optimizers": self.optimizers,
    #            "L": self.L,
    #            "q": self.q,
    #            "batch_size": self.batch_size,
    #            "latent_dim": self.latent_dim,
    #            "z_mean": self.z_mean,
    #            "z_log_var": self.z_log_var,
    #            "_sampling": self._sampling,
    #            "vae": self.vae,
    #            "hist": self.hist
    #            }
    
    # @classmethod
    # def from_config(cls, config):
    #     return cls(**config)

class SVAE(VAE):
    def __init__(self, **kwargs):
        super(SVAE, self).__init__(**kwargs)
        self._is_graph_network=False

    def instantiate_model(self, L, q, latent_dim, batch_size, activation, inner_dim):
        self.inner_dim = inner_dim
        super(SVAE, self).instantiate_model(L, q, latent_dim, batch_size, activation)

    def _enc_dec_layers(self, L, q, latent_dim, batch_size, activation):
        enc_layers = [layers.Flatten(),
                      layers.Dense(self.inner_dim, activation=activation),
                      layers.Dropout(0.3),
                      layers.Dense(self.inner_dim, activation=activation),
                      layers.BatchNormalization(),
                      layers.Dense(self.inner_dim, activation=activation)]

        dec_layers = [layers.Dense(self.inner_dim, activation=activation),
                      layers.Dense(self.inner_dim, activation=activation),
                      layers.Dropout(0.3),
                      layers.Dense(self.inner_dim, activation=activation),
                      layers.Dense(L * q, activation='sigmoid'),
                      ]

        return enc_layers, dec_layers

    def get_config(self):
        return{"optimizer": self.optimizer,
                "METRIC": self.METRIC,
                "N": self.N,               
                "save_root_dir": self.save_root_dir,
                "optimizers": self.optimizers,
                "L": self.L,
                "q": self.q,
                "batch_size": self.batch_size,
                "latent_dim": self.latent_dim,
                "z_mean": self.z_mean,
                "z_log_var": self.z_log_var,
                "_sampling": self._sampling,
                "vae": self.vae,
                "hist": self.hist
                }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
