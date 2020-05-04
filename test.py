import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from dataset import *
import tensorflow as tf

sun_glasses = cv2.imread(r'images/sunglasses.png', cv2.IMREAD_UNCHANGED)

rescale = Rescale((224, 224))
normalize = Normalize()
img = {"image": mpimg.imread(r'data\Bewerbungsphoto.jpg'), "keypoints": np.zeros(136).reshape((-1, 2))}
sample = rescale(img)
test_img = normalize(sample)
test_img = np.stack(test_img['image']).reshape((1, 224, 224, 1))

model = tf.keras.models.load_model('saved_models/face_landmark')
predicted_key_pts = model.predict(test_img)
predicted_key_pts = predicted_key_pts*50+100
predicted_key_pts = predicted_key_pts.reshape((-1, 2)).astype('int')

# show_keypoints(np.squeeze(sample['image']), predicted_key_pts)
# plt.show()
image = np.squeeze(sample['image'])
x = int(predicted_key_pts[17, 0])
y = int(predicted_key_pts[17, 1])

h = int(abs(predicted_key_pts[27, 1]-predicted_key_pts[34, 1]))
w = int(abs(predicted_key_pts[17, 0]- predicted_key_pts[26,0]))

new_sunglasses = cv2.resize(sun_glasses, (w, h), interpolation=cv2.INTER_CUBIC)
roi_color = image[y:y+h, x:x+w]

ind = np.argwhere(new_sunglasses[:,:,3]>0)

for i in range(3):
    roi_color[ind[:,0], ind[:, 1], i] = new_sunglasses[ind[:,0], ind[:, 1], i]
image[y:y+h, x:x+w] = roi_color

plt.imshow(image)
plt.show()
