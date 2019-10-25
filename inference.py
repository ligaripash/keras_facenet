import numpy as np
import os
import matplotlib.pyplot as plt
from imageio import imread
from skimage.transform import resize
import pickle
import cv2
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import distance
from keras.models import load_model
from aggregate_model import avg_pool_model
from IJB_B import get_ijb_b_faces
from PIL import Image
from PIL import ImageOps
from random import sample

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output




def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(cascade_path)

    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=3)
        (x, y, w, h) = faces[0]
        cropped = img[y - margin // 2:y + h + margin // 2,
                  x - margin // 2:x + w + margin // 2, :]
        aligned = resize(cropped, (image_size, image_size), mode='reflect')
        aligned_images.append(aligned)

    return np.array(aligned_images)





def calc_embs(filepaths, margin=10, batch_size=3):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs





def calc_dist(img_name0, img_name1):
    return distance.euclidean(data[img_name0]['emb'], data[img_name1]['emb'])


def calc_dist_plot(img_name0, img_name1):
    print(calc_dist(img_name0, img_name1))
    plt.subplot(1, 2, 1)
    plt.imshow(imread(data[img_name0]['image_filepath']))
    plt.subplot(1, 2, 2)
    plt.imshow(imread(data[img_name1]['image_filepath']))







# data = {}
# for name in names:
#     image_dirpath = image_dir_basepath + name
#     image_filepaths = [os.path.join(image_dirpath, f) for f in os.listdir(image_dirpath)]
#     embs = calc_embs(image_filepaths)
#     # for i in range(len(image_filepaths)):
#     data['{}'.format(name)] = {'name': name,
#                                    'emb': embs}

def create_faces_per_template(ijbb_faces):
    faces_per_template = {}

    for face in ijbb_faces:
        if face.template_id not in faces_per_template.keys():
            faces_per_template[face.template_id] = []
        faces_per_template[face.template_id].append(face)

    return faces_per_template

##############################################################################################################

def prepare_face(face):

    # First read the face image

    image = Image.open(face.image_path)
    rect = face.bb.copy()
    width = rect[2]
    height = rect[3]
    rect[2] += rect[0]
    rect[3] += rect[1]
    output_size = (160, 160)
    cropped_image = image.crop(rect)
    output_image = Image.new('RGB', output_size)
    max_dim = max(rect[2], rect[3])
    if height > width:
        factor = 160 / height
        new_width = int(width * factor)
        resized_image = cropped_image.resize((new_width, 160))
        output_image.paste(resized_image, ( (160 - new_width) // 2,0))

    else:
        factor = 160 / width
        new_height = int(height * factor)
        resized_image = cropped_image.resize((160, new_height))
        output_image.paste(resized_image, ( 0, (160 - new_height) // 2))

    return output_image




##############################################################################################################

def prepare_template_images(faces):
    template_images = []
    template = np.zeros((len(faces), 160, 160, 3))
    for i, face in enumerate(faces):
        image = prepare_face(face)
        # gil - fixed standartization
        np_image = np.array(image)
        np_image = (np_image - 127.5) / 128.0
        # image = prewhiten(np.array(image))
        template[i,:,:,:] = np_image
        # template_images.append(np.array(image))
    return template

##############################################################################################################

def create_pairs_table(faces_per_template, embedding_per_template, ijb_pairs_file):
    pairs_table = []
    with open(ijb_pairs_file) as f:
        for i, line in enumerate(f):
            pairs = line.strip().split(',')
            p_0 = int(pairs[0])
            p_1 = int(pairs[1])
            try:
                face_0 = faces_per_template[p_0][0].subject_id
                face_1 = faces_per_template[p_1][0].subject_id
            except:
                continue
                # print(sys.exc_info(), face_0, face_1)
            try:
                emb_p_0 = embedding_per_template[p_0]
                emb_p_1 = embedding_per_template[p_1]
            except:
                continue
            # if not ok1 or not ok2:
            #     print('skipped pair ', i)
            #     continue
            inner_product = np.inner(emb_p_0, emb_p_1)
            pairs_table.append((p_0, p_1, face_0 == face_1, inner_product))

    with open('pair_table.pickle', 'wb') as f:
        pickle.dump(pairs_table, f)

    return pairs_table


##############################################################################################################

########################################################################################################################

def take_third(elem):
    return elem[3]

########################################################################################################################

def computeROC(pairs_table):
    sorted_table = sorted(pairs_table, key=take_third)
    computed_match = np.zeros((len(sorted_table)))
    gt = np.array([int(item[2]) for item in sorted_table])
    total_gt_positive = np.sum(gt)
    total_gt_negative = len(gt) - total_gt_positive
    nrof_items = float(len(gt))
    TACount = 0
    FACount = 0
    FAR = []
    TAR = []
    print('sorted_table length ', len(sorted_table))
    for i in range(len(sorted_table)-1, len(sorted_table)-1000000, -100):
        j = len(sorted_table) - i
        computed_match[-j:] = 1
        TACount = 0
        FACount = 0
        ta_vec = np.logical_and(gt, computed_match)
        fa_vec = np.logical_and(np.logical_not(gt), computed_match)
        TACount = np.sum(ta_vec)
        FACount = np.sum(fa_vec)
        FAR.append(float(FACount) / total_gt_negative)
        TAR.append(float(TACount) / total_gt_positive)
        if len(FAR) % 100 == 0:
            print(len(FAR))
    return FAR, TAR


if __name__ == '__main__':
    ijb_pairs_file = '/mnt/dsi_vol1/users/gilsh/data/IJB-B/IJB-B/protocol/ijbb_11_S1_S2_matches.csv'
    ijb_b_root_dir = '/mnt/dsi_vol1/users/gilsh/data/IJB-B/IJB-B'
    cascade_path = './model/cv2/haarcascade' \
                   '_frontalface_alt2.xml'
    image_dir_basepath = './data/images/'
    names = ['LarryPage', 'MarkZuckerberg', 'BillGates', 'LarryPage2']
    image_size = 160
    model_path = './model/keras/model/facenet_keras.h5'

    if True:
        with open('pair_table.pickle', 'rb') as f:
            pairs_table = pickle.load(f)
            FAR, TAR = computeROC(pairs_table)
            exit

    model = load_model(model_path)
    aggregate_model = avg_pool_model(model)
    model = aggregate_model

    ijbb_faces = get_ijb_b_faces(ijb_b_root_dir)
    # Create a dictionary between a template id and all the facees it contains
    faces_per_template = create_faces_per_template(ijbb_faces)

    # sample 1000 templates
    keys = faces_per_template.keys()
    sampled_faces_per_template = {}
    # sample_keys = sample(keys, 1000)
    sample_keys = keys
    for key in sample_keys:
        sampled_faces_per_template[key] = faces_per_template[key]

    # For each template read its images, normalize them and calculate its embedding:
    embedding_per_template = {}
    for template_id, template_faces in sampled_faces_per_template.items():
        template_images = prepare_template_images(template_faces)
        template_embedding = model.predict_on_batch(template_images)
        template_embedding = template_embedding / np.linalg.norm(template_embedding)
        embedding_per_template[template_id] = template_embedding
    pass

    with open('embedding_per_template.pickle', 'wb') as f:
        pickle.dump(embedding_per_template, f)
    pass

    pair_table = create_pairs_table(faces_per_template, embedding_per_template, ijb_pairs_file)
    pass
# calc_dist_plot('BillGates0', 'BillGates1')
#
#
#
# calc_dist_plot('MarkZuckerberg0', 'MarkZuckerberg1')
# calc_dist_plot('MarkZuckerberg1', 'MarkZuckerberg2')
#
#
#
#
# X = []
# for v in data.values():
#     X.append(v['emb'])
# pca = PCA(n_components=3).fit(X)
#
# X_BillGates = []
# X_LarryPage = []
# X_MarkZuckerberg = []
# for k, v in data.items():
#     if 'Bill' in k:
#         X_BillGates.append(v['emb'])
#     elif 'Larry' in k:
#         X_LarryPage.append(v['emb'])
#     elif 'Mark' in k:
#         X_MarkZuckerberg.append(v['emb'])
#
# Xd_BillGates = pca.transform(X_BillGates)
# Xd_LarryPage = pca.transform(X_LarryPage)
# Xd_MarkZuckerberg = pca.transform(X_MarkZuckerberg)
#
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# plt.rcParams['legend.fontsize'] = 10
# ax.plot(Xd_BillGates[:, 0], Xd_BillGates[:, 1], Xd_BillGates[:, 2],
#         'o', markersize=8, color='blue', alpha=0.5, label='BillGates')
# ax.plot(Xd_LarryPage[:, 0], Xd_LarryPage[:, 1], Xd_LarryPage[:, 2],
#         'o', markersize=8, color='red', alpha=0.5, label='LarryPage')
# ax.plot(Xd_MarkZuckerberg[:, 0], Xd_MarkZuckerberg[:, 1], Xd_MarkZuckerberg[:, 2],
#         'o', markersize=8, color='green', alpha=0.5, label='MarkZuckerberg')
#
# plt.title('Embedding Vector')
# ax.legend(loc='upper right')

# plt.show()


