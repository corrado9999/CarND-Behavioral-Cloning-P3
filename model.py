import os
import collections
import ast
import numpy as np
import pandas as pd
import scipy.misc
import skimage.transform, skimage.color
import sklearn.model_selection
import tensorflow as tf
import keras as K
import keras.backend
import keras.applications
import keras.preprocessing.image
import click

LOG_FILENAME = 'driving_log.csv'
IMAGE_COLUMNS = 'center left right'.split()
LOG_COLUMNS = IMAGE_COLUMNS + 'steering throttle break speed'.split()
OUTPUT_COLUMNS = 'center center_image left left_image right right_image steering'.split()

# Expose some Tensorflow functions as Keras ===================================
K.backend.name_scope = tf.name_scope
K.backend.constant = tf.constant

def crelu(x):
    # Local import, because otherwise drive.py will fail the model restoration
    import tensorflow as tf
    return tf.nn.crelu(x)

def get_crelu_output_shape(input_shape):
    return tuple(input_shape[:-1]) + (input_shape[-1]*2,)

# New function for preprocessing ==============================================
def rgb2yuv(rgb):
    # Local import, because otherwise drive.py will fail the model restoration
    import numpy as np
    from keras.backend import dot, floatx
    from tensorflow import constant
    Y_WEIGHTS = np.array([0.299, 0.587, 0.114])
    U_WEIGHTS = np.array([0.5 * ((j==2)*1 - Y_WEIGHTS[j]) / (1 - Y_WEIGHTS[2]) for j in range(3)])
    V_WEIGHTS = np.array([0.5 * ((j==0)*1 - Y_WEIGHTS[j]) / (1 - Y_WEIGHTS[0]) for j in range(3)])
    RGB2YUV_MATRIX = constant(np.stack([Y_WEIGHTS, U_WEIGHTS, V_WEIGHTS]).T, dtype=floatx())
    RGB2YUV_BIAS = constant(np.array([-0.5, 0, 0]), dtype=floatx())

    return dot(rgb, RGB2YUV_MATRIX) - RGB2YUV_BIAS

# Dataset loading and cleaning ================================================
def get_dataset(*paths, subset=OUTPUT_COLUMNS):
    print("Loading data")
    full_log = pd.DataFrame(columns=subset)
    for path in paths:
        filename = os.path.join(path, LOG_FILENAME)
        if not os.path.exists(filename):
            print("Skipping non-existing file %r" % filename)
            continue
        log = pd.read_csv(filename, names=LOG_COLUMNS)
        prefix = os.path.commonprefix(log[IMAGE_COLUMNS[0]]
                                      .str.rsplit(os.sep, 2)
                                      .str[0]
                                      .values.tolist())
        if prefix.endswith('IMG'):
            prefix = prefix[:-3].strip()
        for x in set(subset) & set(IMAGE_COLUMNS):
            log[x] = log[x].str.strip().str.replace(prefix, path.strip(), 1)
            log[x + '_image'] = log[x].apply(scipy.misc.imread)

        full_log = full_log.append(log[subset], ignore_index=True)

    full_log.index = full_log['center'].str.split('center_').str[-1].str[:-4].str.split('_').apply(
        lambda x: pd.Timestamp(**dict(zip('year month day hour minute second microsecond'.split(),
                                          tuple(map(int, x[:-1])) + (int(x[-1])*1000,)))))

    return full_log

def reduce_zeros(dataset, factor=4./3, inplace=False):
    h,b = np.histogram(dataset['steering'], bins=25, range=[-1,1])
    z = np.searchsorted(b, 0)
    n = int(h[z-1] - max(h[:z-1].max(), h[z:].max())*factor)
    return dataset.drop(dataset.loc[(dataset['steering']>b[z-1]) &
                                    (dataset['steering']<b[z])]
                               .sample(n).index,
                        axis='rows',
                        inplace=inplace)

# Build model =================================================================
def build_model(input_shape, loss='mse', optimizer='adam'):
    cropping = [(70,24), (0,0)]
    print("Creating network (cropping=%r)" % (cropping,))
    model = K.models.Sequential()
    with K.backend.name_scope('preprocessing'):
        model.add(K.layers.Cropping2D(cropping=cropping, input_shape=input_shape, name='Cropping'))
        model.add(K.layers.Lambda(rgb2yuv, name='rgb2yuv'))
    with K.backend.name_scope('convolutional'):
        model.add(K.layers.Convolution2D(12, 5, 5, name='Conv1',                   subsample=(2,2)))
        model.add(K.layers.Lambda(crelu, name='CReLU', output_shape=get_crelu_output_shape))
        model.add(K.layers.Convolution2D(36, 5, 5, name='Conv2', activation='elu', subsample=(2,2)))
        model.add(K.layers.Convolution2D(48, 5, 5, name='Conv3', activation='elu', subsample=(2,2)))
        model.add(K.layers.Convolution2D(64, 3, 3, name='Conv4', activation='elu'))
        model.add(K.layers.Convolution2D(64, 3, 3, name='Conv5', activation='elu'))
        model.add(K.layers.Dropout(0.5, name='Dropout-0.5'))
    with K.backend.name_scope('top'):
        model.add(K.layers.Flatten(name='Flatten'))
        model.add(K.layers.Dense( 512, name='Dense1', activation='elu'))
        model.add(K.layers.Dense( 128, name='Dense2', activation='elu'))
        model.add(K.layers.Dense(  32, name='Dense3', activation='elu'))
        model.add(K.layers.Dense(   8, name='Dense4', activation='elu'))
        model.add(K.layers.Dense(   1, name='Dense5', activation='softsign'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

# Image generator for data augmentation =======================================
def generate_data(data, batch_size=128, shift=2, rotation=5,
                  images=['center_image', 'left_image', 'right_image'],
                  corrections=[0, 0.2, -0.2],
                  **_
):
    shift, rotation = map(abs, [shift, rotation])
    images, corrections = np.broadcast_arrays(images, corrections)
    indices = np.arange(len(data))
    while True:
        indices = sklearn.utils.shuffle(indices)
        for i in range(0, len(data), batch_size):
            batch = data.iloc[indices[i:i+batch_size]]
            n = len(batch)
            # randomly select camera, shift, rotation and flip
            side = np.random.random_integers(0, len(images)-1, n)
            sft = np.random.random_integers(-shift, shift, (n, 2))
            rot = np.random.random_integers(-rotation, rotation, n)
            flp = np.random.random_integers(0, 1, n)*2 - 1
            # calculate correction to be applied to the steering angle in order
            # to take into account the applied distortions
            cor = corrections[side] + np.radians(rot) + sft[:,0]*0.002
            # extract batch
            X = np.stack(batch[images].values[np.arange(n),side])
            y = np.stack(batch['steering'])
            # for each image, apply the corresponding random shift, rotation
            # and flip in this order
            X = np.stack([
                skimage.transform.rotate(
                    np.pad(x, [(max(0,ss),-min(0,ss)) for ss in s] + [(0,0)],
                              'constant')
                    .__getitem__([slice(-min(0,ss), -max(0,ss) or None)
                                  for ss in s]),
				    r)
                    [:,::f,:]
                for x,r,s,f in zip(X, rot, sft, flp)
            ])
            y = flp*(y + cor)
            yield X, y

@click.command()
@click.argument('input_paths',       nargs=-1)
@click.option('-o', '--output-path', default='model.h5', help='Output file name')
@click.option('-b', '--batch-size',  default=128,        help='The batch size')
@click.option('-e', '--epochs',      default=10,         help='The number of epochs')
@click.option('-L', '--log-dir',     default='',         help='Tensorboard directory')
@click.option('-B', '--save-best',   is_flag=True,       help='Save model when improves validation')
@click.option('-s', '--smooth',      default='',         help='Time duration for a moving window '
                                                              'averaging on the steering angle '
                                                              '(for smoothing)')
@click.option('-v', '--verbose',     count=True,         help='Repeat to increas the verbosity '
                                                              'level for keras.Model.fit (0, 1, '
                                                              'or 2 times)')
@click.option('-p', '--params',      default='{}',       help='Dictionary with other parameters')
def main(input_paths, name='test', output_path='model.h5', batch_size=128, epochs=10, smooth=0,
         log_dir=None, save_best=False, params='{}', verbose=0):
    # Params =================================================================
    params = ast.literal_eval(params)

    # Dataset ================================================================
    dataset = get_dataset(*input_paths)
    if smooth:
        dataset = dataset.sort_index()
        dataset['steering'] = dataset['steering'].rolling("%s" % smooth).mean()
    training_dataset, validation_dataset = sklearn.model_selection.train_test_split(dataset)
    reduce_zeros(training_dataset, inplace=True)

    # Model ==================================================================
    model = build_model(dataset['center_image'].iloc[0].shape)
    callbacks = []
    if log_dir:
        callbacks.append(K.callbacks.TensorBoard(log_dir))
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(output_path,
                                                     verbose=1,
                                                     save_best_only=True))

    # Training ===============================================================
    try:
        history = model.fit_generator(
            generate_data(training_dataset, batch_size, **params),
            len(training_dataset),
            validation_data=generate_data(validation_dataset, batch_size,
                                          **dict(params, images=['center_image'],
                                                         corrections=[0])),
            nb_val_samples=len(validation_dataset),
            nb_epoch=epochs,
            callbacks=callbacks,
            verbose=verbose,
        )
    except KeyboardInterrupt:
        pass

if __name__=='__main__':
    import gc
    main()
    gc.collect()
