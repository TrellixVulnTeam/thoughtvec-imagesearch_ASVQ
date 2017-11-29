import configparser
from models.tensorflow_oxfordnet import vgg19, utils

from urllib.request import urlretrieve
import os
import tensorflow as tf
import numpy as np

class ImageEvaluator():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('.models')
        self.model_params_url = config.get('vgg19', 'params_url')

        self.model_params_dir = 'models/tensorflow_oxfordnet/'
        self.model_params_file = 'vgg19.npy'
        self.model = None
        self.input_ = None

    def download_params(self):
        if not os.path.exists(self.model_params_dir):
            os.makedirs(self.model_params_dir)

        model_params_fp = os.path.join(self.model_params_dir, self.model_params_file)

        if not os.path.isfile(model_params_fp):
            urlretrieve(self.model_params_url, model_params_fp)

    def materialize(self):
        if self.model is None:
            self.download_params()
            self.model = vgg19.Vgg19()
            self.input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
            with tf.name_scope("content_vgg"):
                self.model.build(self.input_)

    def get_formatted_image(self, image_fp):
        img = utils.load_image(image_fp)
        img = img[:, :, 0:3]
        img = np.expand_dims(img, axis=0)
        return img

    def evaluate(self, image_fp):
        with tf.Session() as sess:
            if self.model is None:
                self.materialize()

            img = self.get_formatted_image(image_fp)
            batch = [img]

            images = np.concatenate(batch)
            imgvec = sess.run(self.model.relu6, feed_dict={self.input_: images})
            return imgvec

    def evaluate_batch(self, batch_of_image_fp):
        batch = []
        with tf.Session() as sess:
            if self.model is None:
                self.materialize()

            for image_fp in batch_of_image_fp:
                img = self.get_formatted_image(image_fp)
                batch.append(img)

            images = np.concatenate(batch)
            batch_of_imgvec = sess.run(self.model.relu6, feed_dict={self.input_: images})
            return batch_of_imgvec

if __name__=='__main__':

    fp1 = '/Users/lucaslingle/git/thoughtvec-imagesearch/test_image/dog_on_a_bench.png'
    fp2 = '/Users/lucaslingle/git/thoughtvec-imagesearch/test_image/cat.jpg'

    imgevaluator = ImageEvaluator()

    imgvec = imgevaluator.evaluate(fp1)
    print(imgvec)
    imgvec = imgevaluator.evaluate(fp2)
    print(imgvec)

    imgvecs = imgevaluator.evaluate_batch([fp1, fp2])
    print(imgvecs)