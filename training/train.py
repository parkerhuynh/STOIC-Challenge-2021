import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from ctdataset import  DataGenerator
from config import get_config
from tensorflow.keras import models

import efficientnet_3D.tfkeras as efn 
from classification_models_3D.tfkeras import Classifiers  
from tensorflow import keras
import tensorflow as tf


CONFIGFILE = "/opt/train/config/baseline.json"

def get_datasets(data_dir, config, target):
    image_dir = os.path.join(data_dir, "data/mha/")
    reference_path = os.path.join(data_dir, "metadata/reference.csv")
    df = pd.read_csv(reference_path)
    df["image_path"] = df.apply(lambda row: os.path.join(image_dir, str(row["PatientID"]) + ".mha"), axis=1)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=23)
    df_valid_severe = df_valid[df_valid["probCOVID"] == 1]
    
    #df_train = df_train[:4]
    #df_valid = df_valid[:4]
    #df_valid_severe = df_valid_severe[:4]
    train_generator = DataGenerator(df_train, target, config["preprocess_dir"] , image_dir,  batch_size = config["batch_size"], augment = True)
    val_generator = DataGenerator(df_valid,  target, config["preprocess_dir"] , image_dir,  config["batch_size"], augment = False)
    val_generator_severe = DataGenerator(df_valid_severe, target, config["preprocess_dir"] , image_dir,  config["batch_size"], augment = False)
    return train_generator, val_generator, val_generator_severe


def Network(model_name, width=240, height= 240, depth=240):
    #input layer
    inputs = keras.Input((height, width,depth, 1), name='inputs')
    x = keras.layers.Conv3D(3, (3,3,3), 
                                       strides=(1, 1, 1), 
                                       padding='same', 
                                       use_bias=True)(inputs)
    model_3d, _ = Classifiers.get(model_name)
    model_3d =  model_3d(input_shape=(height, width,depth, 3), include_top=False, weights = False)
    x =  model_3d(x)
    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x =  keras.layers.Dropout(0.7)(x)
    outputs = keras.layers.Dense(units=1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs,
                        outputs = outputs)
    return model

class Metric(keras.callbacks.Callback):
    def __init__(self, dataset, model_name, target, artifact_dir):
        super().__init__()
        self.dataset = dataset
        self.model_name = model_name
        self.target = target
        self.artifact_dir = artifact_dir
        self.max_auc = 0
        self.output = {"val_loss": [], "val_auc": [], "val_acc" : []}
        
    def on_epoch_end(self, epoch: int, logs=None):
        result = self.model.evaluate(self.dataset)
        val_loss, val_auc, val_acc = result
        self.output["val_loss"].append(val_loss)
        self.output["val_auc"].append(val_auc)
        self.output["val_acc"].append(val_acc)
        if val_auc >= self.max_auc:
            self.max_auc = val_auc
            self.model.save(f"{self.artifact_dir}/{self.model_name}_{self.target }.h5")
        
def train(config, train_ds, valid_ds, valid_ds_severe, target, artifact_dir):
    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        for model_name in config["models"]:
            
            if target == "proSevere":
                metric = Metric(valid_ds_severe,  model_name,  target, artifact_dir)
            else:
                metric = Metric(valid_ds,  model_name, target, artifact_dir)
            
            print("Creating the model")
            model =  Network(model_name = model_name)
            print(f"Load model: /opt/train/algorithm/{model_name}_{target}.h5")
            model = models.load_model(f"/opt/train/algorithm/{model_name}_{target}.h5")
            print("Done")
            auc = tf.keras.metrics.AUC(name='auc')
            optimizer = tf.keras.optimizers.Adam(learning_rate = config["lr"])
            
            model.compile(optimizer=optimizer, 
                          metrics=[auc, "acc"],
                          loss = "binary_crossentropy")
            
            print("Training model")
            history = model.fit(train_ds,
                                epochs= config["epochs"],
                                callbacks = [metric]
                               )
            history = history.history
            history.update(metric.output)
            with open(f'{artifact_dir}/{model_name}_{target}.json' , 'w') as file:
                json.dump(history, file)
def do_learning(data_dir, artifact_dir):
    """
    You can implement your own solution to the STOIC2021 challenge by editing this function.
    :param data_dir: Input directory that the training Docker container has read access to. This directory has the same
        structure as the stoic2021-training S3 bucket (see https://registry.opendata.aws/stoic2021-training/)
    :param artifact_dir: Output directory that, after training has completed, should contain all artifacts (e.g. model
        weights) that the inference Docker container needs. It is recommended to continuously update the contents of
        this directory during training.
    :returns: A list of filenames that are needed for the inference Docker container. These are copied into artifact_dir
        in main.py. If your model already produces all necessary artifacts into artifact_dir, an empty list can be
        returned. Note: To limit the size of your inference Docker container, please make sure to only place files that 
        are necessary for inference into artifact_dir.
    """
    print("Geting config")
    config = get_config(CONFIGFILE)
    
    print("Generating datasets")
    for target in [ "proCOVID", "proSevere"]:
        train_ds, valid_ds, valid_ds_severe  = get_datasets(data_dir, config, target)
        train(config, train_ds, valid_ds, valid_ds_severe, target, artifact_dir)

    artifacts = []
    # If your code does not produce all necessary artifacts for the inference Docker container into artifact_dir, return 
    # their filenames:
    # artifacts = ["/tmp/model_checkpoint.pth", "/tmp/some_other_artifact.json"]
    
    return artifacts
