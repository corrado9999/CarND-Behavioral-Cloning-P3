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

def build_model(input_shape, name='test', weights=None, params=None, loss='mse', optimizer='adam'):
    params = infdict(params or {})
    minsize = dict(
        ResNet50=197,
        VGG16=48,
        VGG19=48,
        InceptionV3=299,
    )
    cropping = [(50,20), (0,0)]
    for i,(shp,crp) in enumerate(zip(input_shape, cropping)):
        if shp - sum(crp) < minsize.get(name, 0):
            cropping[i] = (0,0)
    middle_shape = [shp-sum(crp) for crp,shp in zip(cropping, input_shape)]
    padding = [max(minsize.get(name,0)-shp, 0)//2+1 for shp in middle_shape]
    middle_shape = [shp+pdd*2 for pdd,shp in zip(padding, input_shape)]
    middle_shape.append(input_shape[2])
    print("Cropping=%r Padding=%r" % (cropping, padding))

    print("Creating network %r" % name)
    with K.backend.name_scope('preprocessing'):
        preprocessing = [
            K.layers.Cropping2D(cropping=cropping, input_shape=input_shape),
            K.layers.ZeroPadding2D(padding=padding),
            K.layers.Lambda(lambda x: x/255. - 0.5),
        ]
    if name=='test':
        layers = []
    else:
        try:
            with K.backend.name_scope(name):
                base = getattr(K.applications, name)(include_top=False,
                                                     weights=weights,
                                                     input_shape=middle_shape)
        except AttributeError:
            raise ValueError("Unknown network: %r" % name)
        else:
            #base may contain Merge layers, which must be traversed into
            for x in base.layers:
                for y in getattr(x, 'layers', [x]):
                    for z in getattr(y, 'layers', [y]):
                        z.trainable = False
            layers = [
                K.layers.InputLayer(input_tensor=base.output),
            ]
    with K.backend.name_scope('top'):
        top = [
            K.layers.Flatten(),
            K.layers.Dense( 512, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense( 128, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense(  32, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense(   8, activation='relu'), K.layers.Dropout(0.9),
            K.layers.Dense(1),
        ]
    model = K.models.Sequential(preprocessing + layers + top)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

#def data_augmentation(X_train, y_train, batch_size=128,

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
def main(input_paths, name='test', output_path='model.h5', batch_size=128, epochs=10, smooth=0,
         log_dir=None, params='{}', verbose=0):
    params = ast.literal_eval(params)
    dataset = get_dataset(*input_paths)
    if smooth:
        dataset = dataset.sort_index()
        dataset['steering'] = dataset['steering'].rolling("%s" % smooth).mean()
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        *(sklearn.utils.shuffle(
            np.array(dataset['center_image'].values.tolist()),
            dataset['steering']),
    )
    model = build_model(X_train[0].shape, name)
    if log_dir:
        callbacks = [K.callbacks.TensorBoard(log_dir)]
    else:
        callbacks = []
    print(X_train.shape)
    #history = model.fit(
    #    X_train,
    #    y_train,
    #    batch_size=batch_size,
    history = model.fit_generator(
        K.preprocessing.image.ImageDataGenerator(width_shift_range=2./X_train.shape[2],
                                                 height_shift_range=2./X_train.shape[1]).flow(X_train, y_train),
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
