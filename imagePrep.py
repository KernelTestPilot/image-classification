import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import cv2 
from keras.preprocessing.image import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
import modelSettings

BASE_DIR = 'images/'


# labels - age, gender, ethnicity
image_paths = []
age_labels = []
gender_labels = []
gender_dict = {0:'Male', 1:'Female'}
for filename in os.listdir(BASE_DIR):
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    age = int(temp[0])
    gender = int(temp[1])
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)


df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head()


def extract_features(images):
    features = []
    for image in images:
        img = load_img(image, color_mode="grayscale")
        img = img.resize((128, 128), Image.LANCZOS)
        img_array = np.array(img)
        features.append(img_array)

    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 128, 128, 1)
    return features



X = extract_features(df['image'])
X = X/255.0
y_gender = np.array(df['gender'])
y_age = np.array(df['age'])


#Setting up save file
checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True, verbose=1)
# Create model from modelsettings
model = modelSettings.createModel()
#load the latest checkpoint
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
#run the training
history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=30, validation_split=0.2, callbacks=[cp_callback])
model.save_weights(checkpoint_path.format(epoch=0))

image_index = 3
print("Original Gender:", gender_dict[y_gender[image_index]], "Original Age:", y_age[image_index])
# predict from model
pred = model.predict(X[image_index].reshape(1, 128, 128, 1))
pred_gender = gender_dict[round(pred[0][0][0])]
pred_age = round(pred[1][0][0])
print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
plt.axis('off')
plt.imshow(X[image_index].reshape(128, 128), cmap='gray')