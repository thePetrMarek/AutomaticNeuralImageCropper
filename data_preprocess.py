import json
import random

import os

from shutil import copyfile


class DataPreprocess:
    def preprocess(self, folder, validation_size, testing_size):
        preprocessed = []
        with open(folder + "/" + folder + ".json", "r", encoding="utf-8") as file:
            dataset = json.load(file)
            random.shuffle(dataset)
            bins = list(self.chunks(dataset, 50))
            for bin in bins:
                bin.sort(key=lambda x: x["edge_liked_by"]["count"], reverse=True)
                good_examples, bad_examples = self.split_list(bin)
                for good_example in good_examples:
                    preprocessed.append({"picture": good_example["thumbnail_src"].split("/")[-1], "good_example": True})
                for bad_example in bad_examples:
                    preprocessed.append({"picture": bad_example["thumbnail_src"].split("/")[-1], "good_example": False})

        training_set = preprocessed[validation_size + testing_size:]
        with open(folder + "/" + folder + "_training.json", "w", encoding="utf-8") as out_file:
            json.dump(training_set, out_file, indent=4, sort_keys=True)
        print("Created training set of size " + str(len(training_set)))

        validation_set = preprocessed[:validation_size]
        with open(folder + "/" + folder + "_validation.json", "w", encoding="utf-8") as out_file:
            json.dump(validation_set, out_file, indent=4, sort_keys=True)
        print("Created validation set of size " + str(len(validation_set)))

        testing_set = preprocessed[validation_size:validation_size + testing_size]
        with open(folder + "/" + folder + "_testing.json", "w", encoding="utf-8") as out_file:
            json.dump(testing_set, out_file, indent=4, sort_keys=True)
        print("Created testing set of size " + str(len(testing_set)))

    def chunks(self, l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def split_list(self, a_list):
        half = int(len(a_list) / 2)
        return a_list[:half], a_list[half:]

    def merge_topics(self, folder, topics, name_of_merged):
        os.mkdir(name_of_merged)
        merged_json = []
        for topic in topics:
            print("Merging " + topic)
            with open(folder + "/" + topic + "/" + topic + ".json", encoding="UTF-8") as in_file:
                loaded_json = json.load(in_file)
                for image_meta in loaded_json:
                    image_name = image_meta["thumbnail_src"].split("/")[-1]
                    if os.path.isfile(folder + "/" + topic + "/" + image_name):
                        merged_json.append(image_meta)
                        copyfile(folder + "/" + topic + "/" + image_name, name_of_merged + "/" + image_name)
        with open(name_of_merged + "/" + name_of_merged + ".json", "w", encoding="utf-8") as out_file:
            json.dump(merged_json, out_file)


MERGE = True
MAIN_FOLDER = "path/to/folder/with/folders/of/pictures"
IMAGE_SUB_FOLDERS = ["topicOne", "topicTwo", "topicThree"]
MERGED_FOLDER = "merged_topic_folder_name"
VALIDATION_SIZE = 1000
TESTING_SIZE = 1000

dataPreprocess = DataPreprocess()
if MERGE:
    dataPreprocess.merge_topics(MAIN_FOLDER, IMAGE_SUB_FOLDERS, MERGED_FOLDER)
dataPreprocess.preprocess(MERGED_FOLDER, VALIDATION_SIZE, TESTING_SIZE)
