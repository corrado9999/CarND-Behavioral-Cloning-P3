import os
import numpy as np
import pandas as pd
import cv2
import sklearn.model_selection
import tensorflow as tf
import keras as K

LOG_FILENAME = 'driving_log.csv'
IMAGE_COLUMNS = 'center left right'.split()
LOG_COLUMNS = IMAGE_COLUMNS + 'steering throttle break speed'.split()
OUTPUT_COLUMNS = 'center center_image steering'.split()


def get_dataset(*paths, subset=OUTPUT_COLUMNS):
    full_log = pd.DataFrame(columns=subset)
    for path in paths:
        log = pd.read_csv(os.path.join(path, LOG_FILENAME), names=LOG_COLUMNS)
        prefix = os.path.commonprefix(log[IMAGE_COLUMNS[0]]
                                      .str.rsplit(os.sep, 2)
                                      .str[0]
                                      .values.tolist())
        for x in set(subset) & set(IMAGE_COLUMNS):
            log[x] = log[x].str.replace(prefix, path, 1)
            log[x + '_image'] = log[x].apply(cv2.imread)

        full_log = full_log.append(log[subset], ignore_index=True)
    return full_log

def build_model(input_shape, name='test', loss='mse', optimizer='adam'):
    print("Creating network %r" % name)
    if name=='test':
        model = K.models.Sequential([
            K.layers.Flatten(input_shape=input_shape),
            K.layers.Dense(1),
        ])
        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    else:
        raise NameError("Unknown network: %r" % name)
    return model

import click
@click.command()
@click.argument('input_paths',       nargs=-1)
@click.option('-n', '--name',        default='test', help='Name of the Network to use')
@click.option('-o', '--output_path', default='model.h5', help='Output file name')
@click.option('-b', '--batch_size',  default=128, help='The batch size')
@click.option('.e', '--epochs',      default=10, help='The number of epochs')
@click.option('-v', '--verbose',     count=True, help='Repeat to increas the verbosity level for '
                                                      'keras.Model.fit (0, 1 or 2 times)')
def main(input_paths, name='test', output_path='model.h5', batch_size=128, epochs=10, verbose=0):
    dataset = get_dataset(*input_paths)
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        np.array(dataset['center_image'].values.tolist()),
        dataset['steering'],
        shuffle=True
    )
    model = build_model(X_train[0].shape, name)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        nb_epoch=epochs,
        verbose=verbose,
    )
    model.save(output_path)

if __name__=='__main__':
    import gc
    main()
    gc.collect()
