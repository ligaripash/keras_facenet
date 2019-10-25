
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard
from keras.callbacks import LearningRateScheduler
import yaml
import vggface2
from vggface2 import DataGenerator
from keras import backend as K
from inception_resnet_v1 import InceptionResNetV1

## Load parameters
with open('./config_orig.yml', 'r') as f:
    config_params = yaml.load(f)

train_ids, validation_ids, labels = vggface2.create_data_partitions(config_params)

size = config_params['image_size']
num_classes = 10
batch_size = 8
train_dir = config_params['train_dir']
epochs = config_params['epochs']
model_depth = config_params['model_depth']
model_file = config_params['model_file']

params = {'dim': (size,size,3),
          'batch_size': batch_size,
          'shuffle': True,
          'n_classes': num_classes,
          'train_dir': train_dir}


training_generator = DataGenerator(train_ids, labels, **params)
validation_generator = DataGenerator(validation_ids, labels, **params)


sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)

def lr_scheduler(epoch):
    if epoch % 30 == 0:
        K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * 0.1)
    return K.eval(model.optimizer.lr)

change_lr = LearningRateScheduler(lr_scheduler)

weights_path = './model/keras/facenet_keras_weights_orig.h5'
model = InceptionResNetV1( weights_path=weights_path)

model.compile(
    optimizer=sgd
    , loss='categorical_crossentropy'
    , metrics=['accuracy'])


checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tb = TensorBoard(log_dir='./log', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq=100)

callbacks = [checkpoint, tb]

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    epochs=epochs,
                    callbacks=callbacks
                    )


pass
