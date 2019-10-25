

from keras.utils import plot_model
from keras import Model
from keras.layers import Lambda
from keras import backend as K
def avg_pool_model(facenet_model):
    """
    insert a global average pool layer to avg pool all faces in a given template
    :param facenet_model:
    :return:
    """
    # facenet_model.layers.pop()
    # facenet_model.layers.pop()
    # facenet_model.layers.pop()
    # facenet_model.summary()
    average = Lambda(lambda z: K.mean(z, axis=0))
    average.name = 'lambda_99999'
    x = average(facenet_model.layers[-1].output)
    # x = K.map_fn(average, facenet_model.layers[-1].output)
    aggregate_model = Model(inputs=facenet_model.input, outputs=x)
    aggregate_model.summary()

    return aggregate_model
    # plot_model(aggregate_model, to_file='aggregate_model.png')
    # x = GlobalAveragePooling2D()(facenet_model.layers[-1].output)