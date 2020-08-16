import tensorflow as tf
from tensorflow import keras
import numpy as np
import six
from sys import stderr
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
from custom_tqdm import TqdmNotebookCallback

class AdiposeModel(keras.Model):
    def __init__(self, inputs, model_function):
        """
        Because of numerical stability, softmax layer should be
        taken out, and use it only when not training.
        Args
            inputs : keras.Input
            model_function : function that takes keras.Input and returns
            output tensor of logits
        """
        super().__init__()
        outputs = model_function(inputs)
        self.logits = keras.Model(inputs=inputs, outputs=outputs)
        self.logits.summary()
        
    def call(self, inputs, training=None):
        casted = tf.cast(inputs, tf.float32) / 255.0
        if training:
            return self.logits(inputs, training=training)
        return tf.math.sigmoid(self.logits(inputs, training=training))

def run_training(
        model_f, 
        lr_f, 
        name, 
        epochs, 
        batch_size, 
        X_train, 
        Y_train, 
        val_data,
    ):
    """
    val_data : (X_val, Y_val) tuple
    """
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    
    st = time.time()

    inputs = keras.Input((200,200,3))
    mymodel = AdiposeModel(inputs, model_f)
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    mymodel.compile(
        optimizer='adam',
        loss=loss,
        metrics=[
            keras.metrics.BinaryAccuracy(threshold=0.1),
        ]
    )

    logdir = 'logs/fit/' + name
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir,
        histogram_freq=1,
        profile_batch='3,5',
        update_freq='epoch'
    )
    lr_callback = keras.callbacks.LearningRateScheduler(lr_f, verbose=1)

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        verbose=1
    )

    tqdm_callback = TqdmNotebookCallback()

    mymodel.fit(
        x=X_train,
        y=Y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            tensorboard_callback,
            lr_callback,
            save_callback,
            tqdm_callback,
        ],
        verbose=0,
        validation_data=val_data
    )

    print('Took {} seconds'.format(time.time()-st))