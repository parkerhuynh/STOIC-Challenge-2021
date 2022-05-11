import SimpleITK as sitk
import os
import numpy as np
from algorithm.preprocess import preprocess
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import random
from scipy import ndimage
import pickle as pkl

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angle
        angles = list(range(-90, 90, 5))
        angles.remove(0)
        # pick angles at random
        angle = random.choice(angles)
        
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset, target, preprocessed_dir, image_dir, batch_size = 4, augment = False):
        self.dataset = dataset
        self.preprocessed_dir =  preprocessed_dir
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.augment = augment
        self.target = target
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = int(index)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.dataset.iloc[k] for k in indexes]
        images, labels= self.__data_generation(batch_data)
        return images, labels

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
    def __data_generation(self, batch_data):
        images = []
        labels= []
        for image_name, label_1, label_2, image_path  in batch_data:
            preprocessed_foder = os.path.join(self.preprocessed_dir, "preprocess")
            os.makedirs(preprocessed_foder, exist_ok=True)
            preprocessed_path = os.path.join(preprocessed_foder, str(image_name) + ".pkl")
            
            if not os.path.exists(preprocessed_path):
                sitk_image = sitk.ReadImage(image_path)
                image = preprocess(sitk_image)
                print(preprocessed_path)
                fileObject = open(preprocessed_path, 'wb')
                pkl.dump(image, fileObject)
                fileObject.close()
            with open(preprocessed_path, 'rb') as f:
                image = pkl.load(f)
            if  self.augment == True:
                image = rotate(image)
            images.append(image)
            if self.target == "proCOVID":
                labels.append(label_1)
            elif self.target == "proSevere":
                labels.append(label_2)
        return np.array(images),  np.array(labels)

if __name__ == "__main__":
    import sys
    import glob
    import matplotlib.pyplot as plt

    preprocess_dir = sys.argv[1]
    image_dir = sys.argv[2]

    data = [
        {'x': filename, 'y': [0, 0]}
        for filename in glob.glob(os.path.join(image_dir, "*.mha"))
    ]
    ctdataset = CTDataset(data, preprocess_dir)

    steps = 4
    for x, y in ctdataset:
        print(y)
        print(x.shape)
        x = x.numpy()[0]
        length = x.shape[1]
        start = length // 3
        stop = (length // 5) * 4
        step = (stop - start) // steps
        fig, axes = plt.subplots(1, steps, figsize=(15, 4))
        its = range(start, stop, step)
        for it, axis in zip(its, axes):
            screenshot = x[:, it, :][::-1]
            axis.imshow(screenshot, cmap='gray')
            axis.axis('off')
        plt.suptitle(f'label: {y}')
        plt.show()
