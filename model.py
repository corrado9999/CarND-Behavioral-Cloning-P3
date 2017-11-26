import os
import collections
import ast
import numpy as np
import pandas as pd
import scipy.misc
import sklearn.model_selection
import tensorflow as tf
import keras as K
import keras.applications
import keras.preprocessing.image
import click

LOG_FILENAME = 'driving_log.csv'
IMAGE_COLUMNS = 'center left right'.split()
LOG_COLUMNS = IMAGE_COLUMNS + 'steering throttle break speed'.split()
OUTPUT_COLUMNS = 'center center_image steering'.split()

MINSIZE = dict(
    test=(1, 1, 1, 1),
    ResNet50=(1, 197, 197, 1),
    VGG16=(1, 48, 48, 1),
    VGG19=(1, 48, 48, 1),
    InceptionV3=(1, 299, 299, 1),
)

K.backend.name_scope = tf.name_scope
class infdict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(infdict, *args, **kwargs)


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
        for x in set(subset) & set(IMAGE_COLUMNS):
            log[x] = log[x].str.replace(prefix, path, 1)
            log[x + '_image'] = log[x].apply(scipy.misc.imread)

        full_log = full_log.append(log[subset], ignore_index=True)

    full_log.index = full_log['center'].str.split('center_').str[-1].str[:-4].str.split('_').apply(
        lambda x: pd.Timestamp(**dict(zip('year month day hour minute second microsecond'.split(),
                                          tuple(map(int, x[:-1])) + (int(x[-1])*1000,)))))

    # add flipped images
    full_log = pd.concat([
        full_log,
        pd.DataFrame(dict(center=full_log['center'],
                          center_image=tuple(np.stack(full_log['center_image'].values)[...,::-1,:]),
                          steering=-full_log['steering']))
    ])
    return full_log

def max_drift(y_true, y_pred):
    y_true = y_true/180*np.pi
    y_pred = y_pred/180*np.pi
    cos_drift = tf.cumsum(K.backend.cos(y_true)) - tf.cumsum(K.backend.cos(y_pred))
    sin_drift = tf.cumsum(K.backend.sin(y_true)) - tf.cumsum(K.backend.sin(y_pred))
    abs_drift = K.backend.sqrt(K.backend.square(cos_drift) + K.backend.square(sin_drift))
    return K.backend.max(abs_drift)

def build_bottleneck(input_shape, name='test', weights=None):
    layers, _ = get_layers(input_shape, name=name, weights=weights)
    model = K.models.Sequential(layers)
    return model

def build_model(input_shape, name='test', weights=None, loss='mse', optimizer='adam'):
    layers, top = get_layers(input_shape, name=name, weights=weights)
    model = K.models.Sequential(layers + top)
    model.compile(loss=loss, optimizer=optimizer, metrics=[max_drift])
    return model

def get_layers(input_shape, name='test', weights=None):
    print("Creating network %r" % name)
    if name=='test':
        layers = [K.layers.InputLayer(input_shape=input_shape)]
    else:
        try:
            with K.backend.name_scope(name):
                base = getattr(K.applications, name)(include_top=False,
                                                     weights=weights,
                                                     input_shape=input_shape)
        except AttributeError:
            raise ValueError("Unknown network: %r" % name)
        else:
            #base may contain Merge layers, which must be traversed into
            for x in base.layers:
                for y in getattr(x, 'layers', [x]):
                    for z in getattr(y, 'layers', [y]):
                        z.trainable = False
            layers = base.layers
    with K.backend.name_scope('top'):
        top = [
            K.layers.Flatten(),
            K.layers.Dense( 512, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense( 128, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense(  32, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense(   8, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense(1),
        ]
    return layers, top

def kronecker(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim < y.ndim:
        x = x.reshape((1,)*(y.ndim-x.ndim) + x.shape)
    if y.ndim < x.ndim:
        y = y.reshape((1,)*(x.ndim-y.ndim) + y.shape)
    x_subs = 'abcdefghijklmnopqrstuvwxyz'[:x.ndim]
    y_subs = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:y.ndim]
    return np.einsum(
        '{x},{y}->{xy}'.format(x=x_subs, y=y_subs,
                               xy=''.join(a+b for a,b in zip(x_subs, y_subs))),
        x, y).reshape(np.array(x.shape)*y.shape)

def preprocess(X, minsize, batch_size=None, square=False):
    X = X[:,50:-20,:,:]
    repeats = np.ceil(minsize / np.array(X.shape))
    if square:
        repeats[:] = repeats.max()
    print("Resampling: %r" % (repeats,))
    if not batch_size:
        return kronecker(X, np.ones(repeats.astype(int)))
    for i in range(0, len(X), batch_size):
        yield kronecker(X[i:i+batch_size], np.ones(repeats.astype(int)))

def preprocess_batches(batches, minsize, square=False):
    for batch in batches:
        preprocess(batch, minsize, square)

@click.command()
@click.argument('input_paths',       nargs=-1)
@click.option('-n', '--name',        default='test',     help='Name of the Network to use')
@click.option('-o', '--output_path', default='model.h5', help='Output file name')
@click.option('-b', '--batch_size',  default=128,        help='The batch size')
@click.option('-e', '--epochs',      default=10,         help='The number of epochs')
@click.option('-l', '--log-dir',     default='',         help='Tensorboard directory')
@click.option('-s', '--smooth',      default='',         help='Time duration for a moving window '
                                                              'averaging on the steering angle '
                                                              '(for smoothing)')
@click.option('-v', '--verbose',     count=True,         help='Repeat to increas the verbosity '
                                                              'level for keras.Model.fit (0, 1 or '
                                                              '2 times)')
@click.option('-p', '--params',      default='{}',       help='Dictionary with other parameters')
@click.option('-B', '--bottlenecks', default=None,       help='Bottlenecks file name (where '
                                                              'intermediate constant layers are '
                                                              'saved to or read from)')
def main(input_paths, name='test', output_path='model.h5', batch_size=128, epochs=10, smooth=0,
         log_dir=None, params='{}', bottlenecks=None, verbose=0):
    weights = 'imagenet'
    params = ast.literal_eval(params)
    dataset = get_dataset(*input_paths)
    if smooth:
        dataset = dataset.sort_index()
        dataset['steering'] = dataset['steering'].rolling("%s" % smooth).mean()

    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        np.array(dataset['center_image'].values.tolist()),
        dataset['steering'],
    )
    if log_dir:
        callbacks = [K.callbacks.TensorBoard(log_dir)]
    else:
        callbacks = []
    generator = K.preprocessing.image.ImageDataGenerator(width_shift_range=2./X_train.shape[2],
                                                         height_shift_range=2./X_train.shape[1])
    if bottlenecks:
        try:
            with np.load(bottlenecks) as data:
                X_train = data['train']
                X_valid = data['valid']
        except:
            print("Cannot read bottlenecks, computing them from scratch")
            model = build_bottleneck(X_train[0].shape, name, weights=weights)
            X_train = model.predict_generator(preprocess(X_train, MINSIZE[name], batch_size=batch_size),
                                              batch_size=batch_size, len(X_train), verbose=verbose)
            X_valid = model.predict_generator(preprocess(X_train, MINSIZE[name], batch_size=batch_size),
                                              batch_size=batch_size, len(X_valid), verbose=verbose)
            np.save(bottlenecks, train=X_train, valid=X_valid)
        model = build_model(X_train[0].shape, 'test')
    else:
        model = build_model(X_train[0].shape, name, weights=weights)
    #history = model.fit(
    #    X_train,
    #    y_train,
    #    batch_size=batch_size,
    history = model.fit_generator(
        generator.flow(X_train, y_train, batch_size=batch_size),
        len(X_train),
        validation_data=(X_valid, y_valid),
        nb_epoch=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    model.save(output_path)

if __name__=='__main__':
    import gc
    main()
    gc.collect()
