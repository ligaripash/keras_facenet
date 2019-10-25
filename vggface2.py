import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
import keras
import random

# (x_train, y_train), (x_test, y_test) = vggface2.load_data(config_params)


def create_set_partitions(config_params):
    """
    Create list of samples per face and the associated labels for each list
    :param config_params:
    :return:
    list of face lists and the labels for both train and validation
    """
    list_file = config_params['train_list_path']
    all_data = {}
    # gil - take first 100 for debugging
    # all_ids = open(list_file).readlines()[0:2309]
    all_ids = open(list_file).readlines()
    image_per_template = {}
    class_id = -1
    old_template_id = -1
    for id in all_ids:

        template_id = int(id.split('/')[0][1:]) # skip the 'n' prefix
        if template_id != old_template_id:
            class_id += 1
            image_per_template[class_id] = []

        image_per_template[class_id].append(id)

        old_template_id = template_id

    training_pairs = []
    for class_id in image_per_template:
        for f in image_per_template[class_id]:
            training_pairs.append((class_id, f))
    return image_per_template, training_pairs

################################################################################################

def prepare_training_data(image_per_template, train_dir):

    data_size = 0
    for l in image_per_template:
        data_size += len(image_per_template[l])

    X = np.empty((data_size, 160, 160, 3))
    y = np.zeros((data_size,8), dtype=int)
    # y = identity - 2
    # Generate data
    i = 0
    c_id = 0
    for t_id in image_per_template:
        for f_path in image_per_template[t_id]:
        # Store sample
            image_full_path = (train_dir + '/' + f_path).strip()
            im = Image.open(image_full_path)
            im = im_resize_preserve_aspect_ratio(im, 160)

            im_array = np.array(im)
            im_array = (im_array - 127.5) / 128.0

            X[i,] = im_array
            y[i,] = keras.utils.to_categorical(c_id, num_classes=8)
            i += 1
        c_id += 1
        # Store class - the label in vggface2 starts with n0002
        # y[i] = identity - 2

    return X, y


################################################################################################


class SetGenerator(keras.utils.Sequence):
    def __init__(self, images_per_template, training_pairs, num_classes=8632, number_batches_per_epoch=50000,
                 set_size=8, train_dir='', inference_mode=False):
        random.shuffle(training_pairs)
        self.training_pairs = training_pairs
        self.images_per_template = images_per_template
        self.num_classes = num_classes
        self.number_batches_per_epoch = number_batches_per_epoch
        self.set_size = set_size
        self.train_dir = train_dir
        self.inference_mode = inference_mode
        # self.X, self.y = prepare_training_data(images_per_template, train_dir)
        self.batch_size = 32
        self.number_batches_per_epoch = int(len(training_pairs) / self.batch_size)


    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.number_batches_per_epoch

    def create_batch(self, index):
        X = np.empty((self.batch_size, 160,160,3))
        y = np.zeros((self.batch_size, self.num_classes), dtype=int)

        for i, pairs in enumerate(self.training_pairs[index: index + self.batch_size]):
            image_full_path = (self.train_dir + '/' + pairs[1]).strip()
            im = Image.open(image_full_path)
            im = self.im_resize_preserve_aspect_ratio(im, 160)
            im = self.augment_image(im)

            im_array = np.array(im)
            im_array = (im_array - 127.5) / 128.0

            X[i,] = im_array
            y[i,] = keras.utils.to_categorical(pairs[0], num_classes=self.num_classes)
        return X, y

    def __getitem__(self, index):

        # batch_X = self.X[index:index + self.batch_size,]
        # batch_y = self.y[index:index + self.batch_size,]
        #
        # return batch_X, batch_y

        batch_x, batch_y = self.create_batch(index)
        return batch_x, batch_y
        'Generate one batch of data'

        # Generate indexes of the batch
        # Random sample an identity:
        class_index = random.randint(0, len(self.images_per_template) - 1)

        # Sample a set of faces from the chosen identity
        random_identity = list(self.images_per_template.keys())[class_index]

        list_of_sampled_faces = random.sample(self.images_per_template[random_identity], self.set_size)
        # Generate data
        X, y = self.data_generation(list_of_sampled_faces, class_index)
        if self.inference_mode:
            return X, y
        else:
            return X, y

    def augment_image(self, im):
        if random.random() < .5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        rotation_range = 20
        im = im.rotate(random.randint(-rotation_range, rotation_range))
        return im



    def data_generation(self, list_of_sampled_faces, class_index):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.set_size, 160,160,3))
        y = np.zeros((self.set_size,8), dtype=int) + class_index
        # y = identity - 2
        # Generate data
        for i, ID in enumerate(list_of_sampled_faces):
            # Store sample
            image_full_path = (self.train_dir + '/' + ID).strip()
            im = Image.open(image_full_path)
            im = self.im_resize_preserve_aspect_ratio(im, 160)
            im = self.augment_image(im)

            im_array = np.array(im)
            im_array = (im_array - 127.5) / 128.0

            X[i,] = im_array
            y[i,] = keras.utils.to_categorical(class_index, num_classes=self.num_classes)
            # Store class - the label in vggface2 starts with n0002
            # y[i] = identity - 2

        return X, y

    def im_resize_preserve_aspect_ratio(self, im, new_size):
        current_image_size = im.size
        # We scale the larger side of the image to the target size
        max_index = np.argmax(current_image_size)
        ratio = new_size / current_image_size[max_index]


        new_size_same_aspect_ratio = [int(x * ratio) for x in current_image_size]
        rescaled_input_image = im.resize(new_size_same_aspect_ratio)
        shift = int((new_size - min(new_size_same_aspect_ratio)) / 2)
        left_up_roi = [0,0]
        min_index = np.argmin(new_size_same_aspect_ratio)
        left_up_roi[min_index] = shift
        output_image = Image.new("RGB", (new_size, new_size))
        output_image.paste(rescaled_input_image, left_up_roi)
        return output_image

################################################################################################


def create_data_partitions(config_params):
    list_file = config_params['train_list_path']
    all_data = {}
    # gil - take first 100 for debugging
    all_ids = open(list_file).readlines()
    # gil - remove shuffle for speed - remove later
    # random.shuffle(all_ids)
    train_fraction = 0.9
    train_ids = all_ids[0:int(0.9 * len(all_ids))]
    validation_ids = all_ids[int(0.9 * len(all_ids)):]
    labels = {}
    for id in all_ids:
        l = id.split('/')[0]
        labels[id] = l

    return train_ids, validation_ids, labels

################################################################################################

def im_resize_preserve_aspect_ratio(im, new_size):
    current_image_size = im.size
    # We scale the larger side of the image to the target size
    max_index = np.argmax(current_image_size)
    ratio = new_size / current_image_size[max_index]


    new_size_same_aspect_ratio = [int(x * ratio) for x in current_image_size]
    rescaled_input_image = im.resize(new_size_same_aspect_ratio)
    shift = int((new_size - min(new_size_same_aspect_ratio)) / 2)
    left_up_roi = [0,0]
    min_index = np.argmin(new_size_same_aspect_ratio)
    left_up_roi[min_index] = shift
    output_image = Image.new("RGB", (new_size, new_size))
    output_image.paste(rescaled_input_image, left_up_roi)
    return output_image


################################################################################################


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, train_dir, batch_size=32, dim=(32,32,32),
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.train_dir = train_dir

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def augment_image(self, im):
        if random.random() < .5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        rotation_range = 20
        im = im.rotate(random.randint(-rotation_range, rotation_range))
        return im
        # Image degradations:
        # count = 0
        # if random.random() < .1:
        #     im = im.filter(ImageFilter.GaussianBlur(random.randrange(6,16)))

        # pass








    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            image_full_path = (self.train_dir + '/' + ID).strip()
            im = Image.open(image_full_path)
            im = im_resize_preserve_aspect_ratio(im, self.dim[0])
            im = self.augment_image(im)

            im_array = np.asarray(im) / 255.0
            X[i,] = im_array
            # Store class - the label in vggface2 starts with n0002
            y[i] = int(self.labels[ID][1:]) - 2

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)






def load_data(config_params):
    """

    :param config_params:
    :return: (x_train, y_train), (x_test, y_test)
    """
    image_size = config_params['image_size']
    number_of_samples = config_params['number_of_samples']
    x_train = np.empty((number_of_samples, 3, image_size, image_size), dtype='uint8')
    y_train = np.empty((number_of_samples,), dtype='uint8')
    x_train = []
    y_train = []

    # Here is a sample from the file list:
    """
    n000002/0001_01.jpg
    n000002/0002_01.jpg
    n000002/0003_01.jpg
    n000002/0004_01.jpg
    n000002/0005_01.jpg
    n000002/0006_01.jpg
    """
    dir_path = os.path.dirname(config_params['train_list_path'])
    with open(config_params['train_list_path'], 'r') as f:
        for line in f:
            face_id = line.split('/')[0]
            # skip the n
            face_id = int(face_id[1:])
            y_train.append(face_id)
            file_name = os.path.join(dir_path, line)
            im = Image.open(file_name)
            max_size = max(im.size)
            ratio = float(image_size) / float(max_size)
            new_size = tuple([int(x * ratio) for x in im.size])
            im = im.resize(new_size, Image.ANTIALIAS)
            new_im = Image.new("RGB", (image_size, image_size))
            new_im.paste(im, ((image_size-new_size[0])//2,
                    (image_size-new_size[1])//2))
            new_im.show()


