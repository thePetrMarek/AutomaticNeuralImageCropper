import math
from PIL import Image
import numpy as np


class BasicCropper:
    def __init__(self, prediction_probabilities, sess, X):
        self.prediction_probabilities = prediction_probabilities
        self.sess = sess
        self.X = X

    def crop(self, image):
        width, height = image.size
        crop_coordinates = self.make_crop_coordinates(width, height)
        best_crop_coordinate = None
        best_crop_score = 0
        number_of_batches = int(math.ceil(len(crop_coordinates) / 50))
        for i in range(number_of_batches):
            print("Batch " + str(i + 1) + "/" + str(number_of_batches))
            coordinate_batch = crop_coordinates[50 * i: min(50 * (i + 1), len(crop_coordinates))]
            batch = self.make_crops(image, coordinate_batch)
            scores = self.sess.run(self.prediction_probabilities, feed_dict={self.X: batch})[:, 1]

            best_batch_score = max(scores)
            if best_batch_score > best_crop_score:
                best_crop_score = best_batch_score
                best_crop_coordinate = coordinate_batch[np.argmax(scores)]
        [best_crop] = self.make_crops(image, [best_crop_coordinate], False)
        return best_crop, best_batch_score

    def make_crop_coordinates(self, width, height):
        coodinates = []
        shorter_dimension = min(width, height)
        position_step = shorter_dimension / 80
        sizes = [shorter_dimension * 0.8,
                 shorter_dimension * 0.9, shorter_dimension]
        for size in sizes:
            position_width = 0
            while True:
                position_height = 0
                while True:
                    coodinates.append((position_width, position_height,
                                       position_width + size, position_height + size))
                    position_height += position_step
                    if position_height > height - size:
                        break
                position_width += position_step
                if position_width > width - size:
                    break
        return coodinates

    def make_crops(self, image, crop_coordinates, resize=True):
        crops = []
        for crop_coordinate in crop_coordinates:
            crop = image.crop(crop_coordinate)
            if resize:
                crop = crop.resize((299, 299), Image.ANTIALIAS)
            crops.append(np.array(crop))
        return crops
