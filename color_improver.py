from PIL import ImageEnhance, Image
import numpy as np


class ColorImprover:
    def __init__(self, prediction_probabilities, sess, X):
        self.prediction_probabilities = prediction_probabilities
        self.sess = sess
        self.X = X

    def improve(self, image):
        brightness = ImageEnhance.Brightness(image)
        image, score = self.improve_value(brightness)

        contrast = ImageEnhance.Contrast(image)
        image, score = self.improve_value(contrast)

        color = ImageEnhance.Color(image)
        image, score = self.improve_value(color)

        return image, score

    def improve_value(self, enhancer):
        batch = []
        values = []
        for brightness_value in np.arange(0.75, 1.25, 0.01):
            values.append(brightness_value)
            improved_image = enhancer.enhance(brightness_value).resize((299, 299), Image.ANTIALIAS)
            improved_image = np.array(improved_image)
            batch.append(improved_image)
        scores = self.sess.run(self.prediction_probabilities, feed_dict={self.X: batch})[:, 1]
        return enhancer.enhance(values[np.argmax(scores)]), max(scores)
