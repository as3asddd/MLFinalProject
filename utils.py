from PIL import Image
import numpy as np
def data_loader(image_path):
    return np.asarray(Image.open(image_path), dtype=np.float32)

def data_preprocess(image):
    return image / 255

def predict(model, data):
    predict = model.predict(data)
    label = np.argmax(predict)
    return label, predict[0][label]
