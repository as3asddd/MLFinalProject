import keras
import tensorflow as tf
import numpy as np
from utils import *
import sys

if __name__ == "__main__":
    image_path = sys.argv[1]
    image = data_preprocess(data_loader(image_path))
    assert image.shape == (55, 47, 3)
    pca_model = keras.models.load_model("./Models/Fine-Pruning-PCA-Approach/multi_trigger_multi_target_fixed_net.h5")
    pur_model = keras.models.load_model("./Models/Fine-Pruning Approach/GN.h5")
    image = tf.reshape(image, [1, 55, 47, 3])
    pca_label, pca_predict = predict(pca_model, image)
    pur_label, pur_predict = predict(pur_model, image)
    if pca_label == 1283:
      print (1283)
    else:
      print (pur_label if pca_predict < pur_predict else pca_label)
