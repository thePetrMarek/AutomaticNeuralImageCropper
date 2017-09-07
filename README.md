# Automatic Neural Image Cropper
Neural network trained to make the best looking crops of images. It is described in the blog ![Automatic neural image cropper](http://petr-marek.com/blog/2017/09/06/automatic-neural-image-cropper/).

![Automatic Neural Image Cropper](http://petr-marek.com/wp-content/uploads/2017/09/automatic-neural-cropper2-900x506.jpg)

## How to run
1. Download checkpoint of Inception v4 from https://github.com/tensorflow/models/tree/master/slim#pre-trained-models and unpack it to root.
2. Install all requirements by running

    ```pip install requirements.txt```

2. Find some data (you have to do it on your own. AVA dataset maybe...). It is JSON containing array with objects: ```{"picture":"path/to/picture","good_example":True}``` Method to decide if picture is good or bad is described in blog ![Automatic neural image cropper](http://petr-marek.com/blog/2017/09/06/automatic-neural-image-cropper/). You can take inspiration from ![data_preprocess.py](data_preprocess.py).
3. Open ![Main.py](Main.py). There are several parameters at the beginning of the file.

Parameter | Function
--- | --- 
TRAIN | Run training or run cropping (Bool)
EPOCHS | How many epochs will we train? (Int)
BATCH_SIZE | Size of batch (Int)
TRAIN_ACCURACY | Do we want to evaluate accuracy on trainig set? This will make training A LOT slower. (Bool)
MODEL_NAME | Name of model (String)
RESTORE | Will we restore the model from checkpoint? (Bool)
CHECKPOINT | Path to checkpoint (String)
EPOCH | From which epoch to continue? (Int)
[training\|validation\|testing]_dataset | Path to folder with dataset and name of json file (String, String)
IMAGE_FOLDER | Path to folder with photos to crop (String)

4. Set parameters to train and run training by

    ```py -3 Main.py```
    
5. Set parameters to cropping and run cropping by
    
    ```py -3 Main.py```

