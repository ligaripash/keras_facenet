import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import LearningRateScheduler, CSVLogger, ModelCheckpoint, TensorBoard
from keras.callbacks import LearningRateScheduler
import yaml
import vggface2
from vggface2 import SetGenerator
from keras import backend as K
from inception_resnet_v1 import InceptionResNetV1
from keras.models import load_model
from loupe_keras import NetVLAD
from keras.layers import BatchNormalization
## Load parameters
with open('./config.yml', 'r') as f:
    config_params = yaml.load(f)

# train_ids, validation_ids, labels = vggface2.create_set_partitions(config_params)
images_per_template, training_data = vggface2.create_set_partitions(config_params)
num_classes = len(images_per_template)
size = config_params['image_size']
faces_per_template = config_params['faces_per_template']
train_dir = config_params['train_dir']
epochs = config_params['epochs']
experiment_path = config_params['experiment_path']
model_file = config_params['model_file']
max_samples = faces_per_template
# Currently we the batch size is 1 we can consider aggregating from larger batches
batch_size = 1

training_generator = SetGenerator(images_per_template, training_data, set_size=faces_per_template,num_classes=num_classes, number_batches_per_epoch=50000, train_dir=train_dir)
validation_generator = SetGenerator(images_per_template, training_data, set_size=faces_per_template,num_classes=num_classes, number_batches_per_epoch=1000, train_dir=train_dir)
inference_generator = SetGenerator(images_per_template, training_data, set_size=faces_per_template,num_classes=num_classes,
                                   inference_mode=True,number_batches_per_epoch=3, train_dir=train_dir)

# sgd = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
sgd = optimizers.SGD(lr=0.01, nesterov=False)
adam = optimizers.Adam()

# def lr_scheduler(epoch):
#     if epoch % 30 == 0:
#         K.set_value(model.optimizer.lr, K.eval(model.optimizer.lr) * 0.1)
#     return K.eval(model.optimizer.lr)

# change_lr = LearningRateScheduler(lr_scheduler)

facenet_model_path = './model/keras/model/facenet_keras.h5'
facenet_model = load_model(facenet_model_path)
facenet_model.trainable = False
# input x size: (batchsize, max_samples, feature_size)
# output x size: (batchsize, output_dim)
feature_size = config_params['feature_size']
x = facenet_model.layers[-1].output
for l in facenet_model.layers:
    l.trainable=False
# x = keras.layers.Reshape((max_samples,feature_size))(x)

x = facenet_model.layers[-1].output
AVERAGE = True
AGGREGATE = False
if AGGREGATE:
    if AVERAGE:
        average = keras.layers.Lambda(lambda z: K.mean(z, axis=0))
        average.name = 'lambda_average'
        x = average(facenet_model.layers[-1].output)
        x = keras.layers.Lambda(lambda x: K.reshape(x, (1, 512)), name='final_lambda')(x)
    else:
        x = NetVLAD(feature_size=feature_size,
                    max_samples=max_samples,
                    cluster_size=config_params['num_of_clusters'],
                    output_dim=256)(x)

classes = len(images_per_template)

x = keras.layers.Lambda(lambda y: K.l2_normalize(y,axis=1), name="l2_norm")(x)
debug_model = keras.models.Model(inputs=facenet_model.input, outputs=x)
# x = keras.layers.Dense(128, activation='relu', name='dense_before_softmax', kernel_initializer=keras.initializers.Ones())(x)
x = keras.layers.Dense(classes, activation='softmax', name='classification_layer')(x)
# x = keras.layers.Dense(classes,kernel_initializer=keras.initializers.Identity(),
#                        use_bias=False,  name='Bottleneck_final')(x)

# x = BatchNormalization(momentum=0.995, epsilon=0.001, scale=False,
#                        name='final_bn')(x)

full_model = keras.models.Model(inputs=facenet_model.input, outputs=x)

full_model.compile(
    optimizer="adam"
    , loss='categorical_crossentropy'
    , metrics=['accuracy'])

full_model.summary()

checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tb = TensorBoard(log_dir=experiment_path , histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq=100 )

# callbacks = [checkpoint, tb]
callbacks = []

# Debug with small training data
# X, y = vggface2.prepare_training_data(images_per_template, train_dir)
# full_model.fit(X, y, epochs=100, batch_size=1)

# output = debug_model.predict_generator(generator=inference_generator, verbose=0)
# get_3rd_layer_output = K.function([full_model.layers[0].input],
#                                   [model.layers[3].output])
# layer_output = get_3rd_layer_output([x])[0]

full_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1,
                    epochs=epochs,
                    callbacks=callbacks
                    )

pass
