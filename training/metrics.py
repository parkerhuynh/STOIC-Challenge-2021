from tensorflow import keras

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
