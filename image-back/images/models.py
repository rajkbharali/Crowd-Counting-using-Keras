from django.db import models
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, decode_predictions, preprocess_input


# Create your models here.
class Image(models.Model):
    picture = models.ImageField()
    classified = models.CharField(max_length=200, blank=True)
    uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "Image classfied at {}".format(self.uploaded.strftime('%Y-%m-%d %H:%M'))

    def save(self, *args, **kwargs):
        try:
            img = load_img(self.picture, target_size=(480, 640))

            img = np.array(img)
            img = img[np.newaxis, :]
            img = tf.convert_to_tensor(img, dtype=tf.float32)

            model = keras.models.load_model('static/models/final_model2.h5')
            result = model.predict(img)
            self.classified = int(result)
            print('success')
        except Exception as e:
            print('classification failed', e)
        super().save(*args, **kwargs)
