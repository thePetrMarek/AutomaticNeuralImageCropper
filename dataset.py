import json
import random
import numpy as np

from PIL import Image


class Dataset:
    def __init__(self, folder, file):
        self.folder = folder
        with open(folder + "/" + file, "r", encoding="utf-8") as json_file:
            self.dataset = json.load(json_file)
        random.shuffle(self.dataset)
        self.length = len(self.dataset)
        self.index = 0

    def get_batch(self, size):
        inputs = []
        labels = []
        for i in range(size):
            try:
                inputs.append(self.load_picture(self.dataset[self.index]["picture"]))
                label = np.zeros(2)
                label[int(self.dataset[self.index]["good_example"])] = 1
                labels.append(label)
            except OSError as e:
                print("Problem with " + str(self.dataset[self.index]["picture"]))
                print(e)
            self.index += 1
            self.index %= self.length

        return inputs, labels

    def get_original_image(self):
        image_to_return = None
        while image_to_return is None:
            try:
                image_to_return = Image.open(self.folder + "/" + self.dataset[self.index]["picture"]).convert("RGB")
            except OSError as e:
                print("Problem with " + str(self.dataset[self.index]["picture"]))
                print(e)
            self.index += 1
            self.index %= self.length
        return image_to_return

    def load_picture(self, picture):
        image = Image.open(self.folder + "/" + picture).convert("RGB")
        im = image.resize((299, 299), Image.ANTIALIAS)
        im = np.array(im)
        return im
