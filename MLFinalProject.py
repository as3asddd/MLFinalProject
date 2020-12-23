#!/usr/bin/env python
# coding: utf-8

# In[118]:


import tensorflow as tf

from tensorflow import keras
import numpy as np
import argparse
import h5py
from sklearn.decomposition import PCA


# WAR for https://github.com/tensorflow/tensorflow/issues/42728

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# End WAR


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0,2,3,1))

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

def get_acc(model, x, y):
    label = np.argmax(model.predict(x), axis = 1)
    acc = np.mean(np.equal(label, y))
    return acc


# In[119]:


def get_new_model(model, conv_mask_indexs):
    def prune_channel(x, channel_index):
        mask = np.ones(x.shape[-1], dtype = np.float32)
        if len(channel_index) != 0:
            mask[channel_index] = 0
        mask = mask.reshape((1,1,1,-1))
        batch_size = keras.backend.shape(x)[0]
        mask = keras.backend.tile(mask, (batch_size, x.shape[1], x.shape[2], 1))
        return keras.layers.Multiply()([x, mask])
    Input = keras.layers.Input(shape=(55, 47, 3), name = "input")
    Conv1 = keras.layers.Conv2D(filters = 20, kernel_size = 4, activation = "relu", name = "conv_1")(Input)
    Lambda1 = keras.layers.Lambda(prune_channel, arguments = {"channel_index" : conv_mask_indexs["conv_1"]}, name = "lambda_1")(Conv1)
    Maxpooling1 = keras.layers.MaxPooling2D(pool_size=(2,2), strides = (2,2), name = "pool_1")(Lambda1)
    Conv2 = keras.layers.Conv2D(filters = 40, kernel_size = 3, activation = "relu", name = "conv_2")(Maxpooling1)
    Lambda2 = keras.layers.Lambda(prune_channel, arguments = {"channel_index" : conv_mask_indexs["conv_2"]}, name = "lambda_2")(Conv2)
    Maxpooling2 = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2), name = "pool_2")(Lambda2)
    Conv3 = keras.layers.Conv2D(filters = 60, kernel_size = 3, activation = "relu", name = "conv_3")(Maxpooling2)
    Lambda3 = keras.layers.Lambda(prune_channel, arguments = {"channel_index" : conv_mask_indexs["conv_3"]}, name = "lambda_3")(Conv3)
    Maxpooling3 = keras.layers.MaxPooling2D(pool_size = (2, 2), strides = (2,2), name = "pool_3")(Lambda3)
    Conv4 = keras.layers.Conv2D(filters = 80, kernel_size = 2, activation = "relu", name = "conv_4")(Maxpooling3)
    Lambda4 = keras.layers.Lambda(prune_channel, arguments = {"channel_index" : conv_mask_indexs["conv_4"]}, name = "lambda_4")(Conv4)
    Flatten1 = keras.layers.Flatten(name = "flatten_1")(Maxpooling3)
    Flatten2 = keras.layers.Flatten(name = "flatten_2")(Conv4)
    Dense1 = keras.layers.Dense(160, name = "fc_1")(Flatten1)
    Dense2 = keras.layers.Dense(160, name = "fc_2")(Flatten2)
    Add = keras.layers.Add(name = "add_1")([Dense1, Dense2])
    Activation = keras.layers.Activation("relu", name = "activation_1")(Add)
    output_num = 1283
    Dense3 = keras.layers.Dense(output_num, activation = "softmax", name = "output")(Activation)
    new_model = keras.Model(inputs = Input, outputs = Dense3)
    set_weights_name = ["conv_1", "conv_2", "conv_3", "conv_4", "fc_1", "fc_2", "output"]
    for name in set_weights_name:
        new_model.get_layer(name).set_weights(model.get_layer(name).get_weights())
    return new_model


# In[120]:


def get_sorted_conv_mean(model, x_test, y_test, use_conv_names = ["conv_3"]):
    if len(use_conv_names) == 0:
        return []
    layer_outputs = [model.get_layer(layer_name).output for layer_name in use_conv_names]
    output = keras.backend.function(inputs = model.get_layer("input").input, outputs = layer_outputs)(x_test)
    origin_acc = get_acc(model, x_test, y_test)
    conv_outputs_mean = [np.mean(a, axis = 0) for a in output]
    sorted_mean = []
    for i in range(len(conv_outputs_mean)):
        mean_ = np.mean(np.mean(conv_outputs_mean[i], axis = 0), axis = 0)
        tmp = [(use_conv_names[i] + " " + str(j), mean_[j]) for j in range(len(mean_))]
        sorted_mean.extend(tmp)
    sorted_mean = sorted(sorted_mean, key = lambda num : num[1])
    return sorted_mean


# In[121]:


def get_mask_index(model, x_test, y_test, use_conv_names = ["conv_3"], threshold = 0.01):
    conv_mask_indexs = {"conv_1":[],"conv_2":[],"conv_3":[],"conv_4":[]}
    new_model = get_new_model(model, conv_mask_indexs)
    ori_acc = get_acc(model, x_test, y_test)
    new_acc = get_acc(new_model, x_test, y_test)
    sorted_mean = get_sorted_conv_mean(model, x_test, y_test, use_conv_names = use_conv_names)
    left, right = 0, len(sorted_mean)
    count = 1
    mid = 0
    while True:
        conv_mask_tmp_indexs = {"conv_1":[],"conv_2":[],"conv_3":[],"conv_4":[]}
        mid = (left + right) // 2
        mask_means = sorted_mean[0 : mid]
        for mask in mask_means:
            conv_name, channel_index = mask[0].split()
            conv_mask_tmp_indexs[conv_name].append(int(channel_index))
        new_model = get_new_model(model, conv_mask_tmp_indexs)
        new_acc = get_acc(new_model, x_test, y_test)
        print ("Mask iteration {}: Origin Acc:{} New Acc:{} Mask Counts:{}".format(count, ori_acc * 100, new_acc * 100, len(mask_means)))
        if ori_acc - new_acc <= threshold:
            conv_mask_indexs = conv_mask_tmp_indexs
            left = mid
        else:
            right = mid
            
        if right - left <= 1:
            break
        count += 1
    return conv_mask_indexs


# In[123]:


def train_model(model, x_val, y_val,                 do_fine_tunning = True, batch_size = 64, epochs = 50,                 pca_components = 400,
                use_conv_names = ["conv_3"], threshold = 0.01, \
                x_test = None, y_test = None,\
                 x_poi = None, y_poi = None, \
               ):
    
    conv_mask_indexs = get_mask_index(model, x_val, y_val, use_conv_names, threshold)
    new_model = get_new_model(model, conv_mask_indexs)
    print("Before Fine Tunning Val Acc : {}".format(get_acc(new_model, x_val, y_val) * 100))
    if x_test is not None and y_test is not None:
        print ("Before Fine Tunning Test Acc : {}".format(get_acc(new_model, x_test, y_test) * 100))
    if x_poi is not None and y_poi is not None:
        print ("Before Fine Tunning Poi Acc : {}".format(get_acc(new_model, x_poi, y_poi) * 100))
    new_model.compile(
        optimizer=keras.optimizers.Adam(),
        loss = keras.losses.SparseCategoricalCrossentropy(),
        metrics = [keras.metrics.SparseCategoricalAccuracy()],
    )
    if do_fine_tunning:
        fit_history = new_model.fit(
            x_val,
            y_val,
            batch_size = batch_size,
            epochs = epochs,
            validation_data = (x_test, y_test) if (x_test is not None and y_test is not None) else None
        )
    conv_outputs = keras.backend.function(inputs = new_model.get_layer("input").input, outputs = new_model.get_layer("conv_3").output)(x_val)
    #delete_conv_outputs = np.delete(conv_outputs, conv_mask_indexs["conv_3"], axis = -1)
    x_flat_val = conv_outputs.reshape((conv_outputs.shape[0], -1))
    pca = PCA(n_components=pca_components)
    new_val = pca.fit_transform(x_flat_val)
    after_val = np.matmul(new_val, pca.components_)
    T = sorted(np.sum(np.power(after_val - x_flat_val, 2), axis = 1))[-int(0.1 * y_val.shape[0])]
    def pca_transform(x, pca_mean, pca_components, T):
        # WAR for PAC model saving
        pca_mean = np.array(pca_mean, dtype=np.float32)
        pca_components = np.array(pca_components, dtype=np.float32)
        # End WAR
        conv_x = x[0]
        fc_x = x[1]
        ori_x = keras.layers.Flatten()(conv_x)
        mean_ = pca_mean.reshape((1,-1))
        new_x = keras.layers.Add()([ori_x, -1 * mean_])
        new_x = keras.layers.Reshape((1, new_x.shape[1]))(new_x)
        batch_size = keras.backend.shape(new_x)[0]
        comT = pca_components.T
        comT = comT.reshape((-1,) + comT.shape)
        comT = keras.backend.tile(comT, (batch_size,1,1))
        new_x = keras.backend.batch_dot(new_x, comT)
        com = keras.backend.tile(np.expand_dims(pca_components, axis = 0), (batch_size,1,1))
        after_x = keras.backend.batch_dot(new_x, com) 
        after_x = keras.layers.Flatten()(after_x)
        err = keras.backend.sum(keras.backend.pow(after_x - ori_x, 2), axis = 1)
        poi_index = keras.backend.greater_equal(err, T)
        poi_label = tf.where(poi_index == True, np.inf, 0)
        poi_label = keras.layers.Reshape((1,))(poi_label)
        poi_label = keras.backend.cast(poi_label, "float32")
        fc_x = keras.layers.Concatenate(axis = 1)([fc_x, poi_label])
        return fc_x 

    Lambda5 = keras.layers.Lambda(pca_transform, arguments = {"pca_mean" : pca.mean_, "pca_components" : pca.components_, "T":T}, name = "lambda_5")([new_model.get_layer("conv_3").output,new_model.get_layer("output").output])
    new_model = keras.Model(inputs = new_model.get_layer("input").input, outputs = Lambda5)
    new_model.compile()
    print("After Fine Tunning Val Acc : {}".format(get_acc(new_model, x_val, y_val) * 100))
    if x_test is not None and y_test is not None:
        print ("After Fine Tunning Test Acc : {}".format(get_acc(new_model, x_test, y_test) * 100))
    if x_poi is not None and y_poi is not None:
        print ("After Fine Tunning Poi Acc : {}".format(get_acc(new_model, x_poi, y_poi) * 100))


    fc_output = np.argmax(new_model.predict(x_poi))
    return new_model


# In[116]:


def main(args):
    model = keras.models.load_model(args.model)
    clean_val_path = args.val_data
    clean_test_path = args.test_data
    poi_path = args.poisoned_data
    x_val, y_val = data_loader(clean_val_path)
    x_val = data_preprocess(x_val)
    x_test, y_test, x_poi, y_poi = None, None, None, None
    if clean_test_path != None:
        x_test, y_test = data_loader(clean_test_path)
        x_test = data_preprocess(x_test)
    if poi_path != None:
        x_poi, y_poi = data_loader(poi_path)
        x_poi = data_preprocess(x_poi)

    if args.action == "train":
        new_model = train_model(
            model, x_val, y_val,
            do_fine_tunning = args.fine_tunning,
            batch_size=args.batch_size,
            epochs = args.train_epochs,
            use_conv_names = args.pruning_conv_layer.split(";"),
            threshold = args.threshold,
            x_test = x_test, y_test = y_test,
            x_poi = x_poi, y_poi = y_poi,
        )
        if args.save_path is not None:
            new_model.save(args.save_path + ".h5")

    if args.action == "infer":
        new_model = model
        print("Model Val Acc : {}".format(get_acc(new_model, x_val, y_val) * 100))
        if x_test is not None and y_test is not None:
            print ("Model Test Acc : {}".format(get_acc(new_model, x_test, y_test) * 100))
        if x_poi is not None and y_poi is not None:
            print ("Model Poi Acc : {}".format(get_acc(new_model, x_poi, y_poi) * 100))

# In[117]:


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_data", type=str, default = "data/clean_validation_data.h5", help = "input validation data")
    parser.add_argument("--test_data", type=str, default=None, help="for model training test acc")
    parser.add_argument("--threshold", type = float, default = 0.01, help = "pruning acc threshold")
    parser.add_argument("--poisoned_data", type = str, default = None, help = "poison datapath")
    parser.add_argument("--model", type = str, default="models/sunglasses_bd_net.h5", help = "keras model path")
    parser.add_argument("--action", type=str, default = "train", choices = ["infer", "train"], help = "train model or just inference")
    parser.add_argument("--pruning_conv_layer", type=str, default="conv_3", help = "conv layer need to be pruning, split with ;")
    parser.add_argument("--fine_tunning", type=bool, default=True, help = "if using fine_tunning after pruning")
    parser.add_argument("--save_path", type=str, default="fix", help = "fixed model saved path")
    parser.add_argument("--train_epochs", type=int, default=1, help = "train epochs")
    parser.add_argument("--batch_size", type=int, default=64, help = "train batch size")
    parser.add_argument("--pca_path", type=str, default="fix.pkl", help = "saved pca path")
    args = parser.parse_args()
    main(args)


# In[ ]:




