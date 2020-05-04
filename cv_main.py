import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, BatchNormalization, Dropout, Activation
from keras.models import Sequential, Model

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cv2
from dataset import *
from sklearn.utils import shuffle


def show_keypoints(image, key_pts, gt_pts=None):
    plt.imshow(image, cmap='gray')
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        # plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i+1)

        image = test_images[i]
        predicted_key_pts = test_outputs[i]
        predicted_key_pts = predicted_key_pts*50+100
        predicted_key_pts = predicted_key_pts.reshape((-1, 2)).astype('int')

        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts*50 + 100
            ground_truth_pts = ground_truth_pts.reshape((-1, 2)).astype('int')
            show_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        else:
            show_keypoints(np.squeeze(image), predicted_key_pts)
        plt.axis('off')
    plt.show()


def load_data(path):
    key_pts_frame = pd.read_csv(path)
    image_names = key_pts_frame.iloc[:, 0]

    train_dataset = []
    train_label = []

    for idx, name in enumerate(image_names):
        image_name = os.path.join('data/training/', name)
        key_pts = key_pts_frame.iloc[idx, 1:]
        key_pts = key_pts.astype("float").values.reshape((-1, 2))

        sample = {"image": mpimg.imread(image_name), "keypoints": key_pts}
        sample = rescale(sample)
        sample = crop(sample)
        sample = normalize(sample)
        train_dataset.append(sample['image'])
        train_label.append(sample["keypoints"].reshape((-1, 1)))

    train_dataset = np.stack(train_dataset)
    train_label = np.stack(train_label)
    return train_dataset, train_label


def load_data_test(path):
    key_pts_frame = pd.read_csv(path)
    image_names = key_pts_frame.iloc[:, 0]

    train_dataset = []
    train_label = []

    for idx, name in enumerate(image_names):
        image_name = os.path.join('data/test/', name)
        key_pts = key_pts_frame.iloc[idx, 1:]
        key_pts = key_pts.astype("float").values.reshape((-1, 2))

        sample = {"image": mpimg.imread(image_name), "keypoints": key_pts}
        sample = rescale(sample)
        sample = crop(sample)
        sample = normalize(sample)
        train_dataset.append(sample['image'])
        train_label.append(sample["keypoints"].reshape((-1, 1)))

    train_dataset = np.stack(train_dataset)
    train_label = np.stack(train_label)
    return train_dataset, train_label


rescale = Rescale((224, 224))
# crop = RamdonCorp((224, 224))
normalize = Normalize()
# train_dataset, train_label = load_data('data/training_frames_keypoints.csv')
# # train_dataset, train_label = shuffle(train_dataset, train_label)
# train_dataset = train_dataset[..., np.newaxis]
# train_label = train_label.reshape((-1, 136))

# test_dataset, test_label = load_data_test('data/test_frames_keypoints.csv')
# test_dataset, test_label = shuffle(test_dataset, test_label)
# test_dataset = test_dataset[..., np.newaxis]
# test_label = test_label.reshape((-1, 136))
#
model = tf.keras.models.load_model('saved_models/face_landmark')
# new_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
#
test_img = {"image": mpimg.imread(r'data\Bewerbungsphoto.jpg'), "keypoints": np.zeros(136).reshape((-1, 2))}
sample = rescale(test_img)
# sample = crop(sample)
test_img = normalize(sample)
test_img = np.stack(test_img['image']).reshape((1, 224, 224, 1))
#
# feature_maps = new_model(test_img)
# square = 8
# idx = 1
# for i in range(square):
#     for j in range(square):
#         ax = plt.subplot(square, square, idx)
#         plt.imshow(feature_maps[0, :, :, idx-1], cmap='gray')
#         idx+=1
#         plt.axis('off')
# plt.show()

# model.summary()

# weights, bias = model.layers[0].get_weights()
# weights = np.array(weights)
# print(weights.shape)
# for i in range(16):
#     ax = plt.subplot(1, 16, i + 1)
#     weight = weights[:,:,:,i]
#     plt.imshow(weight.reshape((3, 3)), cmap='gray')
#     plt.axis('off')
# plt.show()

# filter_idx = 0
#
# weight = weights[filter_idx]
# print(weight)
# plt.imshow(weight, cmap='gray')
# plt.show()

# test_img = test_dataset[:5]
# test_label = test_label[:5]

# test_img = {"image": mpimg.imread(r'data\Bewerbungsphoto.jpg'), "keypoints": np.zeros(136).reshape((-1, 2))}
# sample = rescale(test_img)
# sample = crop(sample)
# test_img = normalize(sample)
# test_img = np.stack(test_img['image']).reshape((1, 224, 224, 1))
visualize_output(test_img, model.predict(test_img), batch_size=1)

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# model = Sequential([
#     Conv2D(32, (3, 3), input_shape=(224, 224, 1), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.5),
#     Conv2D(64, (5, 5),  strides=(2, 2), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Conv2D(128, (5, 5), strides=(2, 2), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
#     Dropout(0.5),
#     BatchNormalization(),
#     Conv2D(256, (5, 5), strides=(2, 2), activation='relu'),
#     # MaxPooling2D(pool_size=(2, 2)),
#     # BatchNormalization(),
#     Dropout(0.5),
#     Flatten(),
#     Dense(512),
#     Dropout(0.5),
#     Activation('relu'),
#     Dense(512),
#     Dropout(0.5),
#     Activation('relu'),
#     Dense(136)
# ])

# test_img = train_dataset[:1]
# test_label = train_label[:1]
# # visualize_output(test_img, model.predict(test_img), test_label, batch_size=1)
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])
# history = model.fit(train_dataset, train_label, batch_size=64, epochs=200, verbose = 2)
#
# model.save('saved_models/face_landmark')
# #
# plt.subplot(2, 1, 1)
# plt.plot(history.history['acc'])
# # plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper right')
#
# # Plot history for loss
# plt.subplot(2, 1, 2)
# plt.plot(history.history['loss'])
# # plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper right')
# test_num = 602
# sample = face_dataset[test_num]
#
# fig = plt.figure()
# for i, tx in enumerate([rescale, crop]):
#     transformed_sample = tx(sample)
#
#     ax = plt.subplot(1, 2, i+1)
#     ax.set_title(type(tx).__name__)
#     show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])
#
# plt.show()

# plt.savefig(r"saved_models/history")

# order matters! i.e. rescaling should come before a smaller crop
# for i in range(3):
#
#     fig = plt.figure(figsize=(20, 20))
#
#     rand_i = np.random.randint(0, len(face_dataset))
#     sample = face_dataset[rand_i]
#
#     print(i, sample['image'].shape, sample['keypoints'].shape)
#
#     ax = plt.subplot(1, 3, i+1)
#     ax.set_title("sample #{}".format(i))
#     show_keypoints(sample['image'], sample['keypoints'])
#
# plt.show()
# os.system("shutdown /s /t 1")