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

K.backend.name_scope = tf.name_scope
K.backend.constant = tf.constant
class infdict(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(infdict, *args, **kwargs)

def rgb2yuv(rgb):
    import numpy as np
    from keras.backend import dot, floatx
    from tensorflow import constant
    Y_WEIGHTS = np.array([ 0.299,  0.587,  0.114])
    U_WEIGHTS = np.array([0.5 * ((j==2)*1 - Y_WEIGHTS[j]) / (1 - Y_WEIGHTS[2]) for j in range(3)])
    V_WEIGHTS = np.array([0.5 * ((j==0)*1 - Y_WEIGHTS[j]) / (1 - Y_WEIGHTS[0]) for j in range(3)])
    RGB2YUV_MATRIX = constant(np.stack([Y_WEIGHTS,
                                                  U_WEIGHTS,
                                                  V_WEIGHTS]).T, dtype=floatx())
    RGB2YUV_BIAS = constant(np.array([-0.5, 0, 0]), dtype=floatx())

    return dot(rgb, RGB2YUV_MATRIX) - RGB2YUV_BIAS

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

    return full_log

def build_model(input_shape, name='test', weights=None, loss='mse', optimizer='adam'):
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
    elif name.lower()=='nvidia':
        preprocessing[-1] = K.layers.Lambda(rgb2yuv)
        with K.backend.name_scope(name):
            layers = [
                K.layers.Convolution2D(24, 5, 5, activation='elu', subsample=(2,2)),
                K.layers.Convolution2D(36, 5, 5, activation='elu', subsample=(2,2)),
                K.layers.Convolution2D(48, 5, 5, activation='elu', subsample=(2,2)),
                K.layers.Convolution2D(64, 3, 3, activation='elu'),
                K.layers.Convolution2D(64, 3, 3, activation='elu'),
                K.layers.Dropout(0.5),
            ]
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
            K.layers.Dense( 512, activation='elu'),
            K.layers.Dense( 128, activation='elu'),
            K.layers.Dense(  32, activation='elu'),
            K.layers.Dense(   8, activation='elu'),
            K.layers.Dense(1),
        ]
    model = K.models.Sequential(preprocessing + layers + top)
    model.compile(loss=loss, optimizer=optimizer)
    return model

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
            side = np.random.random_integers(0, len(images)-1, n)
            sft = np.random.random_integers(-shift, shift, (n, 2))
            rot = np.random.random_integers(-rotation, rotation, n)
            flp = np.random.random_integers(0, 1, n)*2 - 1
            cor = corrections[side] + np.radians(rot) + sft[:,0]*0.002
            X, y = np.stack(batch[images].values[np.arange(n),side]), np.stack(batch['steering'])
            X = np.pad([skimage.transform.rotate(np.roll(x, s, (0,1))[shift:-shift, shift:-shift, :],
                                                 r)[:,::f,:]
                        for x,r,s,f in zip(X, rot, sft, flp)],
                       [(0,0), (shift,shift), (shift,shift), (0,0)],
                       'constant'
            )
            print(side, sft, rot, flp, cor, y)
            y = flp*(y + cor)
            yield X, y

@click.command()
@click.argument('input_paths',       nargs=-1)
@click.option('-n', '--name',        default='test',     help='Name of the Network to use')
@click.option('-o', '--output_path', default='model.h5', help='Output file name')
@click.option('-b', '--batch_size',  default=128,        help='The batch size')
@click.option('-e', '--epochs',      default=10,         help='The number of epochs')
@click.option('-L', '--log-dir',     default='',         help='Tensorboard directory')
@click.option('-B', '--save-best',   default=False,      help='Save model when improves validation')
@click.option('-s', '--smooth',      default='',         help='Time duration for a moving window '
                                                              'averaging on the steering angle '
                                                              '(for smoothing)')
@click.option('-v', '--verbose',     count=True,         help='Repeat to increas the verbosity '
                                                              'level for keras.Model.fit and '
                                                              'callbacks (0, 1, 2 or 3 times)')
@click.option('-p', '--params',      default='{}',       help='Dictionary with other parameters')
def main(input_paths, name='test', output_path='model.h5', batch_size=128, epochs=10, smooth=0,
         log_dir=None, save_best=False, params='{}', verbose=0):
    # Params =================================================================
    params = ast.literal_eval(params)
    weights = params.pop('weights', 'imagenet')

    # Dataset ================================================================
    dataset = get_dataset(*input_paths)
    if smooth:
        dataset = dataset.sort_index()
        dataset['steering'] = dataset['steering'].rolling("%s" % smooth).mean()
    training_dataset, validation_dataset = sklearn.model_selection.train_test_split(dataset)

    # Model ==================================================================
    model = build_model(dataset['center_image'].iloc[0].shape, name, weights=weights)
    callbacks = []
    if log_dir:
        callbacks.append(K.callbacks.TensorBoard(log_dir))
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(output_path,
                                                     verbose=verbose-1,
                                                     save_best_only=True))

    # Training ===============================================================
    save = True
    try:
        history = model.fit_generator(
            generate_data(training_dataset, batch_size, **params),
            len(training_dataset),
            #validation_data=generate_data(validation_dataset, batch_size,
            #                              **dict(params, images=['center_image'],
            #                                             corrections=[0])),
            #nb_val_samples=len(validation_dataset),
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
