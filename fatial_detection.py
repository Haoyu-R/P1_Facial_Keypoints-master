import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from dataset import *

image = cv2.imread('images/obamas.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()

face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image, 1.2, 2)

image_new = np.copy(image)

# for (x,y,w,h) in faces:
#     cv2.rectangle(image_new, (x, y), (x+w, y+h), (255, 0, 0), 2)
#
# plt.imshow(image_new)
# plt.show()

rescale = Rescale((224, 224))

normalize = Normalize()

model = tf.keras.models.load_model('saved_models/face_landmark')
for (x,y,w,h) in faces:
    roi = image_new[y:y+h,x:x+w]
    test_img = {"image": roi, "keypoints": np.zeros(136).reshape((-1, 2))}
    test_img = rescale(test_img)
    test_img = normalize(test_img)
    test_img = np.stack(test_img['image']).reshape((1, 224, 224, 1))
    visualize_output(test_img, model.predict(test_img), batch_size=1)


