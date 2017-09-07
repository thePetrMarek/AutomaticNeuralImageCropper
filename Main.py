import os

from PIL import Image
from tensorflow.contrib import slim

import inception_utils
import tensorflow as tf

from basic_cropper import BasicCropper
from color_improver import ColorImprover
from dataset import Dataset
from inception_v4 import inception_v4
import scipy.misc

TRAIN = False

EPOCHS = 10
BATCH_SIZE = 50
TRAIN_ACCURACY = False
MODEL_NAME = "baseline"
RESTORE = True
CHECKPOINT = "checkpoints/" + MODEL_NAME + "/baseline-7827"
EPOCH = 0

FOLDER = "image/folder"


def accuracy(dataset, batch_size):
    dataset_length = dataset.length
    accuracy_sum = 0
    for i in range(int(dataset_length / BATCH_SIZE)):
        images, labels = dataset.get_batch(BATCH_SIZE)
        accuracy_sum += sess.run(batch_accuracy, feed_dict={X: images, Y: labels})
    accuracy = accuracy_sum / int(dataset_length / batch_size)
    return accuracy


with tf.Graph().as_default():
    sess = tf.Session()

    im_size = 299
    X = tf.placeholder(tf.float32, (None, im_size, im_size, 3))
    Y = tf.placeholder(tf.int64, shape=[None, 2])

    inception_v4.default_image_size = im_size
    arg_scope = inception_utils.inception_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception_v4(X, is_training=False)
    inception_output = end_points['PreLogitsFlatten']

    saver = tf.train.Saver()

    with tf.name_scope("fully_connected"):
        w = tf.Variable(tf.truncated_normal([1536, 2], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
        prediction_logits = tf.matmul(inception_output, w) + b
        prediction_probabilities = tf.nn.softmax(prediction_logits, name="prediction_probabilities")

    with tf.name_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction_logits,
                                                                name="cross_entropy")
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

    with tf.name_scope("optimalization"):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_operation = optimizer.minimize(loss, var_list=[w, b])

    with tf.name_scope("accuracy"):
        prediction_class = tf.argmax(prediction_logits, 1)
        Y_class = tf.argmax(Y, 1)
        equality = tf.equal(prediction_class, Y_class)
        batch_accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    init_new_vars_op = tf.global_variables_initializer()
    sess.run(init_new_vars_op)
    saver.restore(sess, 'inception_v4.ckpt')

    training_dataset = Dataset("training_dataset", "training_dataset.json")
    validation_dataset = Dataset("validation_dataset", "validation_dataset.json")
    testing_dataset = Dataset("testing_dataset", "testing_dataset.json")

    loader = tf.train.Saver()

    if RESTORE:
        loader.restore(sess, CHECKPOINT)
        starting_epoch = EPOCH
    else:
        starting_epoch = 0

    if TRAIN:
        saver = tf.train.Saver(max_to_keep=4)
        writer = tf.summary.FileWriter("summaries/" + MODEL_NAME)
        writer.add_graph(sess.graph)

        if not os.path.exists(os.path.join("checkpoints", MODEL_NAME)):
            os.makedirs(os.path.join("checkpoints", MODEL_NAME))

        for epoch in range(starting_epoch, EPOCHS):
            print("STARTING EPOCH " + str(epoch + 1) + "/" + str(EPOCHS))
            for i in range(int(training_dataset.length / BATCH_SIZE)):
                learning_step = epoch * int(training_dataset.length / BATCH_SIZE) + i

                images, labels = training_dataset.get_batch(BATCH_SIZE)
                loss_value, _ = sess.run([loss, train_operation], feed_dict={X: images, Y: labels})

                summary = tf.Summary()
                summary.value.add(tag='Loss', simple_value=loss_value)
                writer.add_summary(summary, learning_step)

                if i % 10 == 0:
                    summary = tf.Summary()
                    validation_accuracy = accuracy(validation_dataset, BATCH_SIZE)
                    summary.value.add(tag='Accuracy_validation', simple_value=validation_accuracy)
                    if TRAIN_ACCURACY:
                        train_accuracy = accuracy(training_dataset, BATCH_SIZE)
                        summary.value.add(tag='Accuracy_train', simple_value=train_accuracy)
                    writer.add_summary(summary, learning_step)

                    print("Epoch " + str(epoch + 1) + "/" + str(EPOCHS) + ", BATCH " + str(i + 1) + "/" + str(
                        int(training_dataset.length / BATCH_SIZE)))
                    if TRAIN_ACCURACY:
                        print("Loss " + str(loss_value) + "Training accuracy " + str(
                            train_accuracy) + ", Validation accuracy " + str(validation_accuracy))
                    else:
                        print("Loss " + str(loss_value) + ", Validation accuracy " + str(validation_accuracy))
                        print()

                    saver.save(sess, os.path.join("checkpoints", MODEL_NAME, MODEL_NAME), global_step=learning_step)

        test_accuracy = accuracy(testing_dataset, BATCH_SIZE)
        print("Testing accuracy " + str(test_accuracy))
    else:
        basic_cropper = BasicCropper(prediction_probabilities, sess, X)
        color_improver = ColorImprover(prediction_probabilities, sess, X)
        valid_images = [".jpg", ".png"]
        for f in os.listdir(FOLDER):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            image_path = os.path.join(FOLDER, f)
            print(str(f))
            image = Image.open(image_path).convert("RGB")
            image, score = basic_cropper.crop(image)
            image = Image.fromarray(image)
            image, score = color_improver.improve(image)
            print("Score: " + str(score))
            index = image_path.rfind(".")
            new_image_path = image_path[:index] + "_cropped." + image_path[index + 1:]
            scipy.misc.imsave(new_image_path, image)
