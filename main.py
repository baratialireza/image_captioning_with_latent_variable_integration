# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 15:15:09 2022

@author: SamRayaneh
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:06:26 2022

@author: SamRayaneh
"""


# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.__version__)
# tf.test.is_gpu_available()
print(tf.config.list_physical_devices())

# You'll generate plots of attention in order to see which parts of an image
# your model focuses on during captioning
import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image



"""## Download and prepare the MS-COCO dataset

You will use the [MS-COCO dataset](http://cocodataset.org/#home) to train your model. The dataset contains over 82,000 images, each of which has at least 5 different caption annotations. The code below downloads and extracts the dataset automatically.

**Caution: large download ahead**. You'll use the training set, which is a 13GB file.
"""

annotation_file = 'D:/programs/barati/image captioning/mscoco\dataset/annotations_trainval2014/annotations/captions_train2014.json'
PATH='D:/programs/barati/image captioning/mscoco/dataset/train2014/train2014/'


"""## Optional: limit the size of the training set 
To speed up training for this tutorial, you'll use a subset of 30,000 captions and their corresponding images to train your model. Choosing to use more data would result in improved captioning quality.
"""

with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Group all captions together having the same image ID.
image_path_to_caption = collections.defaultdict(list)
image_id_to_caption = collections.defaultdict(list)
image_id_to_path = collections.defaultdict(list)

for val in annotations['annotations']:
  caption = f"<start> {val['caption']} <end>"
  image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
  image_path_to_caption[image_path].append(caption)
  
  image_id_to_caption[val['image_id']].append(caption)
  image_id_to_path[val['image_id']].append(image_path)


# image_paths = list(image_path_to_caption.keys())
# random.shuffle(image_paths)
image_ids = list(image_id_to_caption.keys())
random.shuffle(image_ids)


# Select the first 6000 image_paths from the shuffled set.
# Approximately each image id has 5 captions associated with it, so that will
# lead to 30,000 examples.
numberofsample=20000

selected_image_ids=image_ids[:numberofsample]
train_image_paths=[]
for i in range(len(selected_image_ids)):
    train_image_paths.append(image_id_to_path[selected_image_ids[i]][0])
    
# train_image_paths = image_ids[:numberofsample]
print(len(train_image_paths))

train_captions = []
img_name_vector = []
for image_path in train_image_paths:
  caption_list = image_path_to_caption[image_path]
  train_captions.extend(caption_list)
  img_name_vector.extend([image_path] * len(caption_list))

print(train_captions[0])
Image.open(img_name_vector[0])

"""## Preprocess the images using InceptionV3
Next, you will use InceptionV3 (which is pretrained on Imagenet) to classify each image. You will extract features from the last convolutional layer.

First, you will convert the images into InceptionV3's expected format by:
* Resizing the image to 299px by 299px
* [Preprocess the images](https://cloud.google.com/tpu/docs/inception-v3-advanced#preprocessing_stage) using the [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/preprocess_input) method to normalize the image so that it contains pixels in the range of -1 to 1, which matches the format of the images used to train InceptionV3.
"""




"""## Preprocess and tokenize the captions

You will transform the text captions into integer sequences using the [TextVectorization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization) layer, with the following steps:

* Use [adapt](https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization#adapt) to iterate over all captions, split the captions into words, and compute a vocabulary of the top 5,000 words (to save memory).
* Tokenize all captions by mapping each word to it's index in the vocabulary. All output sequences will be padded to length 50.
* Create word-to-index and index-to-word mappings to display results.
"""

caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

# We will override the default standardization of TextVectorization to preserve
# "<>" characters, so we preserve the tokens for the <start> and <end>.
def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")

# Max word count for a caption.
max_length = 50
# Use the top 5000 words for a vocabulary.
vocabulary_size = 5000
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)
# Learn the vocabulary from the caption data.
tokenizer.adapt(caption_dataset)

# Create the tokenized vectors
cap_vector = caption_dataset.map(lambda x: tokenizer(x))

# Create mappings for words to indices and indicies to words.
word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)

"""## Split the data into training and testing"""

img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(img_name_vector, cap_vector):
  img_to_cap_vector[img].append(cap)

# Create training and validation sets using an 80-20 split randomly.
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

img_name_train = []
cap_train = []
for imgt in img_name_train_keys:
  capt_len = len(img_to_cap_vector[imgt])
  img_name_train.extend([imgt] * capt_len)
  cap_train.extend(img_to_cap_vector[imgt])

img_name_val = []
cap_val = []
for imgv in img_name_val_keys:
  capv_len = len(img_to_cap_vector[imgv])
  img_name_val.extend([imgv] * capv_len)
  cap_val.extend(img_to_cap_vector[imgv])

len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)

"""## Create a tf.data dataset for training

Your images and captions are ready! Next, let's create a `tf.data` dataset to use for training your model.
"""

# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

# Load the numpy files
def map_func(img_name, cap):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  return img_tensor, cap

# dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))
# # Use map to load the numpy files in parallel
# dataset = dataset.map(lambda item1, item2: tf.numpy_function(
#           map_func, [item1, item2], [tf.float32, tf.int64]),
#           num_parallel_calls=tf.data.AUTOTUNE)

# # Shuffle and batch
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

"""## Model

Fun fact: the decoder below is identical to the one in the example for [Neural Machine Translation with Attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention).

The model architecture is inspired by the [Show, Attend and Tell](https://arxiv.org/pdf/1502.03044.pdf) paper.

* In this example, you extract the features from the lower convolutional layer of InceptionV3 giving us a vector of shape (8, 8, 2048).
* You squash that to a shape of (64, 2048).
* This vector is then passed through the CNN Encoder (which consists of a single Fully connected layer).
* The RNN (here GRU) attends over the image to predict the next word.
"""

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


"""## Try it on your own images

For fun, below you're provided a method you can use to caption your own images with the model you've just trained. Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the training data (so be prepared for weird results!)

"""
"""# Next steps

Congrats! You've just trained an image captioning model with attention. Next, take a look at this example [Neural Machine Translation with Attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention). It uses a similar architecture to translate between Spanish and English sentences. You can also experiment with training the code in this notebook on a different dataset.
"""



##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


"""# Next steps

Congrats! You've just trained an image captioning model with attention. Next, take a look at this example [Neural Machine Translation with Attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention). It uses a similar architecture to translate between Spanish and English sentences. You can also experiment with training the code in this notebook on a different dataset.
"""

annotation_file2 = 'D:/programs/barati/image captioning/mscoco\dataset/annotations_trainval2014/annotations/instances_train2014.json'


# Read the json file
with open(annotation_file2, 'r') as f:
     instances_train2014 = json.load(f)


import collections
# Group all captions together having the same image ID.
image_to_annot = collections.defaultdict(list)
k=[]
for val in instances_train2014['annotations']:
  category_id = val['category_id']
  image_id = val['image_id']
  k.append(image_id)
  image_to_annot[image_id].append(category_id)

# key=np.unique(k) 
key=selected_image_ids


image_to_id = collections.defaultdict(list)
for val in instances_train2014['images']:
  file_name = val['file_name']
  id2 = val['id']
  image_to_id[id2].append(file_name)



dictionary = collections.defaultdict(list)
supercategory= collections.defaultdict(list)
for val in instances_train2014['categories']:
  file_name = val['name']
  id2 = val['id']
  sup_cat=val['supercategory']
  dictionary[id2].append(file_name)
  supercategory[id2].append(sup_cat)


train_annot={}
list_image=[]
train_captions2=[]
train_category=[]

for k in range(len(key)):
    val=key[k]
    a=image_to_annot[val]
    b=image_to_id[val][0]
    list_image.append(b)
    train_annot[b]=a
    c=[]
    d=[]
    for i in range(len(a)):
        c.append(dictionary[a[i]][0])
        d.append(supercategory[a[i]][0])
    train_captions2.append(c)
    train_category.append(d)



sub_cat_name=[]
for k in list(supercategory):
    ss=supercategory[k][0]
    if ss not in sub_cat_name:
        sub_cat_name.append(ss)

sub_cat_sub=[]
for k in list(sub_cat_name):
    b=[]
    for j in list(supercategory):
        if supercategory[j][0]==k:
            b.append(dictionary[j][0])
    sub_cat_sub.append(b)

print(len(train_category))
print(train_category[0])
print(train_captions2[0])

print(len(sub_cat_sub))
print(sub_cat_sub)



from keras.preprocessing import image




train_sub=[]
concept_category_caption=collections.defaultdict(list)
concept_category_features=collections.defaultdict(list)
concept_category_id=collections.defaultdict(list)

ind=[]
for i in  range(len(train_captions2[:numberofsample])):
    b=[]
    b=train_captions2[i]
    c=[]
    for j in range(len(sub_cat_sub)):
        a=sub_cat_sub[j]
        for k in list(b):
            if k in a:
                c.append(j)
        for k in list(a):
            if k in b:
                concept_category_caption[j].append(b)
                #concept_category_features[j].append(X[i])
                concept_category_id[j].append(list_image[i])
                if j==0:
                  ind.append(i)  
                break
    train_sub.append(c)

print('concept categoring is don !')


# ###### for index 0 : person ####################################
# all_ind=[]
# for i in range(numberofsample):
#     all_ind.append(i)

# ind2=[]
# for i in range(numberofsample):
#     if all_ind[i] not in ind:
#         ind2.append(i)

# for  i in range(len(ind2)):
#     concept_category_caption[0].append(train_captions2[ind2[i]])
#     #concept_category_features[0].append(X[ind2[i]])

# del ind, ind2,all_ind




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import keras

train_sub2=[]
for i in range(len(train_sub)):
    train_sub2.append(str(train_sub[i]))


def calc_max_length(tensor):
    return max(len(t) for t in tensor)

top_k = len(sub_cat_sub)+1 
tokenizer_concept_all = tf.keras.preprocessing.text.Tokenizer(num_words=top_k)
tokenizer_concept_all.fit_on_texts(train_sub2)
train_seqs = tokenizer_concept_all.texts_to_sequences(train_sub2)
cap_vector=tokenizer_concept_all.sequences_to_matrix(train_seqs, mode='freq')
max_length = calc_max_length(cap_vector)


###########################################################################
from sklearn.model_selection import train_test_split
from keras.preprocessing import image

# train_image = []
# #len(list_image)
# for i in range(len(train_image_paths)):
#     img = image.load_img(train_image_paths[i],target_size=(200,200,3))
#     img = image.img_to_array(img)
#     img = img/255
# #    image = preprocess_input(image)
#     train_image.append(img)
# X = np.array(train_image)
# del train_image



# X_train, X_val, y_train, y_val = train_test_split(X, cap_vector, random_state=42, test_size=0.1)
# del X,cap_vector



# import time
# start_total = time.time()
# from keras.layers import Dense, Dropout, Flatten

# #from keras_applications.resnet import ResNet101
# from tensorflow.keras.applications import ResNet101

# #InceptionResNetV2
# from tensorflow.keras.optimizers import SGD

# from keras.applications.vgg16 import VGG16
# #import tf.keras.applications.ResNet101
# from keras.models import Model
# from keras.layers import BatchNormalization
# from keras.callbacks import EarlyStopping
# from keras.callbacks import ModelCheckpoint
# from matplotlib import pyplot
# from keras.utils.vis_utils import plot_model

 

# # define cnn model
# in_shape=(200, 200, 3)
# out_shape=max_length
# # load model 
# #base_model =VGG16(weights='imagenet',include_top=False)
# #base_model = InceptionV3(weights='imagenet',include_top=False)
# #base_model = keras.applications.Xception(weights='imagenet',include_top=False)


# base_model = ResNet101(weights='imagenet',include_top=False)

# base_model.trainable = False

# inputs = keras.Input(shape=in_shape)
# x = base_model(inputs)
# x = keras.layers.GlobalAveragePooling2D()(x)
# #class1 = Dense(1024, activation='relu')(x)
# class1 = Dense(out_shape*16, activation='relu')(x)
# class2= Dropout(0.5)(class1)
# #    class2_2=BatchNormalization()(class2)
# #class3 = Dense(256, activation='relu')(class2)
# class3 = Dense(out_shape*8, activation='relu')(class2)
# class4= Dropout(0.5)(class3)
# #    class4_2=BatchNormalization()(class4)
# #    class5 = Dense(64, activation='relu')(class4_2)
# #    class6= Dropout(0.5)(class5)
# outputs  = Dense(out_shape, activation='softmax')(class4)
#     # define new model
# model_concept_all = Model(inputs, outputs)
#     # compile model
# opt = SGD(lr=0.001, momentum=0.9)
# model_concept_all.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_concept_all.summary()
# # model_concept_all.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)
# from keras.preprocessing.image import ImageDataGenerator
# aug = ImageDataGenerator()
# history =model_concept_all.fit_generator(aug.flow(X_train, y_train,batch_size=64), epochs=5, validation_data=(X_val, y_val))


# es = EarlyStopping(monitor="val_loss",
#     min_delta=0,
#     patience=4,
#     verbose=0,
#     mode="auto",
#     baseline=None,
#     restore_best_weights=True)
# base_model.trainable = True
# opt = SGD(lr=0.0001, momentum=0.8)
# model_concept_all.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
# model_concept_all.summary()
# # history =model_concept_all.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=16, callbacks=[es])
# # history =model_concept_all.fit(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=16)

# history =model_concept_all.fit_generator(aug.flow(X_train, y_train,batch_size=64), epochs=100, validation_data=(X_val, y_val), callbacks=[es])

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
PATH2='D:/programs/barati/image captioning/mscoco/attention class vision/italy/result'

# model_concept_all.save(PATH2+'/model_concept_all.h5')
model_concept_all = keras.models.load_model(PATH2+'/model_concept_all.h5')

# plot_model(model_concept_all, to_file=(PATH2+'/model_generalnet.png'), show_shapes=True, show_layer_names=True)
# # plot training history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
# tt_global=(time.time() - start_total)
# print ('Total Time = '+str(tt_global)+' second')


# del X_train, X_val, y_train, y_val
# print('delete X_train, X_val, y_train, y_val')









##################################################################################################



def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

"""## Initialize InceptionV3 and load the pretrained Imagenet weights

Now you'll create a tf.keras model where the output layer is the last convolutional layer in the InceptionV3 architecture. The shape of the output of this layer is ```8x8x2048```. You use the last convolutional layer because you are using attention in this example. You don't perform this initialization during training because it could become a bottleneck.

* You forward each image through the network and store the resulting vector in a dictionary (image_name --> feature_vector).
* After all the images are passed through the network, you save the dictionary to disk.

"""

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

"""## Caching the features extracted from InceptionV3
"""

# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(16)

for img, path in image_dataset:
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())


print('faeture of train data is build')







##################################################################################################
"""### ***گروه بندی داده های ترین و کپشنهایشان به مفاهیم موجود***"""
num_of_cat = len(sub_cat_sub)
PATH='D:/programs/barati/image captioning/mscoco/dataset/train2014/train2014/'

img_name_train_cat=[]
cap_train_cat=[]
img_name_val_cat=[]
cap_val_cat=[]

for i in range(num_of_cat):
    img_to_cap_vector2 = collections.defaultdict(list)
    for j in range(len(concept_category_id[i])):
        img_to_cap_vector2[PATH+concept_category_id[i][j]]=img_to_cap_vector[(PATH+concept_category_id[i][j])]
        
        # Create training and validation sets using an 80-20 split randomly.
        img_keys = list(img_to_cap_vector2.keys())
        random.shuffle(img_keys)
        
        slice_index = int(len(img_keys)*0.8)
        img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

        img_name_train = []
        cap_train = []
        for imgt in img_name_train_keys:
          capt_len = len(img_to_cap_vector2[imgt])
          img_name_train.extend([imgt] * capt_len)
          cap_train.extend(img_to_cap_vector2[imgt])

        img_name_val = []
        cap_val = []
        for imgv in img_name_val_keys:
          capv_len = len(img_to_cap_vector2[imgv])
          img_name_val.extend([imgv] * capv_len)
          cap_val.extend(img_to_cap_vector2[imgv])

        len(img_name_train), len(cap_train), len(img_name_val), len(cap_val)
        
    img_name_train_cat.append( img_name_train)
    cap_train_cat.append(cap_train )
    img_name_val_cat.append(img_name_val)
    cap_val_cat.append(cap_val)
    




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
encoder = collections.defaultdict(list)
decoder = collections.defaultdict(list)
optimizer = collections.defaultdict(list)

for i in range(num_of_cat):
    encoder[i] = CNN_Encoder(embedding_dim)
    decoder[i] = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
    optimizer[i] = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


"""## Checkpoint"""
checkpoint_path=[]
ckpt_manager=collections.defaultdict(list)
for i in range (num_of_cat):
  checkpoint_path.append( "./checkpoints/train"+str(i))
  ckpt = tf.train.Checkpoint(encoder=encoder[i],
                             decoder=decoder[i],
                             optimizer=optimizer[i])
  
  ckpt_manager[i] = tf.train.CheckpointManager(ckpt, checkpoint_path[i], max_to_keep=5)
  start_epoch = 0
  




loss_plot = []

@tf.function
def train_step(img_tensor, target,k):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder[k].reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder[k](img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder[k](dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder[k].trainable_variables + decoder[k].trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer[k].apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dataset_cat=collections.defaultdict(list)

for i in range (num_of_cat):
  dataset = tf.data.Dataset.from_tensor_slices((img_name_train_cat[i], cap_train_cat[i]))
  # Use map to load the numpy files in parallel
  dataset = dataset.map(lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int64]),
            num_parallel_calls=tf.data.AUTOTUNE)
  
  # Shuffle and batch
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  dataset_cat[i]=dataset
 
 


"""# **اعمال شبکه روی هر گروه**"""

print(len(dataset))
print(len(dataset_cat[0]))

EPOCHS =31
tf.config.run_functions_eagerly(True)

loss_plot=collections.defaultdict(list)
save_path=[]

for i in range(num_of_cat):
  print(i)
  for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0


    for (batch, (img_tensor, target)) in enumerate(dataset_cat[i]):
        batch_loss, t_loss = train_step(img_tensor, target,i)
        total_loss += t_loss

        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    # storing the epoch end loss value to plot later
    loss_plot[i].append(total_loss / num_steps)

    if epoch % 5 == 0:
      ckpt_manager[i].save()

    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

print(len(loss_plot[3]))

PATH2='D:/programs/barati/image captioning/mscoco/attention class vision/italy/result/loss plots'

# plt.plot(history.history['loss'], label='train_global')
# plt.plot(history.history['val_loss'], label='test_global')
for i in range(len(loss_plot)):
    plt.plot(loss_plot[i],label = ['subnet_'+str(i)])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.legend(fontsize=5)
    plt.savefig(PATH2+'/plot'+str(i)+'.png', dpi=300, bbox_inches='tight')
plt.show()


# # Save a dictionary into a pickle file.
# import pickle
# pickle.dump(loss_plot, open(PATH2+"/loss_plot.p", "wb"))  # save it into a file named save.p
# -------------------------------------------------------------
# # Load the dictionary back from the pickle file.
# import pickle
# loss_plot = pickle.load(open(PATH2+"/loss_plot.p", "rb"))



print(loss_plot[0][2])
print(loss_plot[1][2])
print(loss_plot[2][2])
print(loss_plot[3][2])
print(loss_plot[4][2])



###############################################################################
def evaluate(image,k):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder[k].reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder[k](img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder[k](dec_input,
                                                         features,
                                                         hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result/2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tf.compat.as_text(index_to_word(i).numpy())
                         for i in cap_val[rid] if i not in [0]])
result, attention_plot = evaluate(image,1)

print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
annotation_file = 'D:/programs/barati/image captioning/mscoco/dataset/annotations_trainval2014/annotations/instances_val2014.json'
PATH='D:/programs/barati/image captioning/mscoco/dataset'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)
image_to_annot = collections.defaultdict(list)
k=[]
for val in annotations['annotations']:
  category_id = val['category_id']
  image_id = val['image_id']
  k.append(image_id)
  image_to_annot[image_id].append(category_id)
print(len(k))
key=np.unique(k) 
print(len(key))
print(key[16600])

annotation_file = 'D:/programs/barati/image captioning/mscoco/dataset/annotations_trainval2014/annotations/captions_val2014.json'
with open(annotation_file, 'r') as f:
    annotations = json.load(f)
caption_val = collections.defaultdict(list)
k=[]
for val in annotations['annotations']:
  capti = val['caption']
  image_id = val['image_id']
  k.append(image_id)
  caption_val[image_id].append(capti)
print(len(k))
key=np.unique(k) 
print(len(key))
print(key[16600])


image_id_val=[]
img_name_val_orginal=[]
for val in annotations['images']:
  image_id_val.append(val['id'])
  img_name_val_orginal.append(PATH+'/val2014/val2014/'+val['file_name'])

print(len(image_id_val))
key2=np.unique(image_id_val) 
print(len(image_id_val))

print(len(key2))
print(key2[3])

print(len(img_name_val_orginal))
print(img_name_val_orginal[3])
print(image_id_val[3])
print(key2[3])



###############################################################################
#%% %%%%%%%%%%%%%%%%%%%%%%%%%% Prediction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from keras.preprocessing import image


number_of_test=len(img_name_val_orginal)#3000
test_image = []
#len(list_image)
for i in range(number_of_test):
    img = image.load_img(img_name_val_orginal[i],target_size=(200,200,3))
    img = image.img_to_array(img)
    img = img/255
#    image = preprocess_input(image)
    test_image.append(img)
# X_test = np.array(test_image)
# del test_image


num_of_concept=5
weight_concept=[]
index_concept_test=[]
actual, predicted = list(), list()
predicted2=list()

#for k in range (len(X)):
for k in range (len(test_image)):
    if k % 100 == 0:
      print(k)
    img=test_image[k]#X_test[k]
#    img=X[k]
    top_3=[]
    proba = model_concept_all.predict(img.reshape(1,200,200,3))
    top_3 =((-proba).argsort())[0][:num_of_concept]
    weight_concept.append(proba[0][top_3])
    result=[]
    for i in range(len(top_3)):
        result.append(tokenizer_concept_all.index_word[round(top_3[i])])
    #a=int(result)
    index_concept_test.append(result)
    
    p=[]
    act=[]

    for j in range(len(result)):
        a=int(result[j])
        
        
        image = img_name_val_orginal[k]
        id=image_id_val[k]
        
        result2, attention_plot = evaluate(image,a)
        
        act.append(caption_val[image_id_val[k]])
        r2=result2[0:(len(result2)-1)]
        r3=' '.join(r2)
        p.append(r2)
    actual.append(act)
    predicted.append(p[0])#p[0]+p[1]+p[2]
    predicted2.append(p)


actual2=[]
for i in range(len(actual)):
    act2=[]
    for j in range(3):
        act=[]
        for k in range(5):
            act.append(actual[i][j][k].split())
        # act2.append(act)
    actual2.append(act)

del actual
actual=actual2
del actual2, act, act2


from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
# calculate BLEU score
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)));
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)));












# ################################# finding best blue score ####################################
res_all=[]
res_all2=[]
res_all2_ind=[]

for k in range(3):
    predicted3=[]
    for i in range(len(predicted)):
        predicted3.append(predicted2[i][k])
    
    res=[]
    res2=[]
    res2_ind=[]
    for i in range(len(actual)):
        b=sentence_bleu(actual[i], predicted3[i], weights=(1.0, 0, 0, 0))
        res.append(b)
        if b>.7:
            res2_ind.append(i)
            res2.append(b)
    res_all.append(res)
    res_all2.append(res2)
    res_all2_ind.append(res2_ind)
    

np.mean(res_all[2])

# ############################### mean on best result for each images #################################
res_selected=[]
for i in range(len(actual)):
        res_selected.append(np.max([res_all[0][i],res_all[1][i],res_all[2][i]]))
np.mean(res_selected)




# ############################# attention plot and show on best blue score #############################
concept_detected=[]
predicted_best_blue=[]
attention_plot_best_blue=[]

for ws in range(len(res_all2_ind)):
    cd=[]
    rd=[]
    ap=[]
    for rs in range(len(res_all2_ind[ws])):
        rid = res_all2_ind[ws][rs]
        
        img=test_image[rid]
        top_3=[]
        proba = model_concept_all.predict(img.reshape(1,200,200,3))
        top_3 =((-proba).argsort())[0][:num_of_concept]
        weight_concept.append(proba[0][top_3])
        result=[]
        for i in range(len(top_3)):
            result.append(tokenizer_concept_all.index_word[round(top_3[i])])
        
        a=int(result[0])
        image = img_name_val_orginal[rid]
        id=image_id_val[rid]
        result2, attention_plot = evaluate(image,a)
        
        rd.append(result2)
        ap.append(attention_plot)
        cd.append(sub_cat_sub[a])
    predicted_best_blue.append(rd)
    attention_plot_best_blue.append(ap)
    concept_detected.append(cd)



wich_s=1
rs= np.random.randint(0, len(res_all2_ind[wich_s]))
rid = res_all2_ind[wich_s][rs]
real_caption = caption_val[image_id_val[rid]] 
print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(predicted_best_blue[wich_s][rs]))
image = img_name_val_orginal[rid]
plot_attention(image, predicted_best_blue[wich_s][rs], attention_plot_best_blue[wich_s][rs])
print('concept detected:', concept_detected[wich_s][rs])

from PIL import Image
image = Image.open(image)
image.show()

################################ save ################################
PATH2='D:/programs/barati/image captioning/mscoco/attention class vision/italy/result/loss plots'
# Save a dictionary into a pickle file.
import pickle
pickle.dump(predicted_best_blue, open(PATH2+"/predicted_best_blue.p", "wb"))
pickle.dump(attention_plot_best_blue, open(PATH2+"/attention_plot_best_blue.p", "wb"))
pickle.dump(concept_detected, open(PATH2+"/concept_detected.p", "wb"))

# -------------------------------------------------------------
# # Load the dictionary back from the pickle file.
# import pickle
# predicted_best_blue = pickle.load(open(PATH2+"/predicted_best_blue.p", "rb"))
# attention_plot_best_blue = pickle.load(open(PATH2+"/attention_plot_best_blue.p", "rb"))
# concept_detected = pickle.load(open(PATH2+"/concept_detected.p", "rb"))



# ################################
# random with change caption availibility
# ################################
wich_s=0
rs= np.random.randint(0, len(res_all2_ind[wich_s]))
rid = res_all2_ind[wich_s][rs]

img=test_image[rid]
top_3=[]
proba = model_concept_all.predict(img.reshape(1,200,200,3))
top_3 =((-proba).argsort())[0][:num_of_concept]
weight_concept.append(proba[0][top_3])
result=[]
for i in range(len(top_3)):
    result.append(tokenizer_concept_all.index_word[round(top_3[i])])

a=int(result[0])
image = img_name_val_orginal[rid]
id=image_id_val[rid]
result2, attention_plot = evaluate(image,a)

real_caption = caption_val[image_id_val[rid]] 
print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result2))
plot_attention(image, result2,attention_plot)
print('concept detected:', sub_cat_sub[a])

from PIL import Image
image = Image.open(image)
image.show()





# ################################ Flickr_8k #########################################

# # load doc into memory
# def load_doc(filename):
# 	# open the file as read only
# 	file = open(filename, 'r')
# 	# read all text
# 	text = file.read()
# 	# close the file
# 	file.close()
# 	return text

# # load a pre-defined list of photo identifiers
# def load_set(filename):
# 	doc = load_doc(filename)
# 	dataset = list()
# 	# process line by line
# 	for line in doc.split('\n'):
# 		# skip empty lines
# 		if len(line) < 1:
# 			continue
# 		# get the image identifier
# 		identifier = line.split('.')[0]
# 		dataset.append(identifier)
# 	return set(dataset)


# # load clean descriptions into memory
# def load_clean_descriptions(filename, dataset):
# 	# load document
# 	doc = load_doc(filename)
# 	descriptions = dict()
# 	for line in doc.split('\n'):
# 		# split line by white space
# 		tokens = line.split()
# 		# split id from description
# 		image_id, image_desc = tokens[0], tokens[1:]
# 		# skip images not in the set
# 		if image_id in dataset:
# 			# create list
# 			if image_id not in descriptions:
# 				descriptions[image_id] = list()
# 			# wrap description in tokens
# 			desc = image_desc
# 			# store
# 			descriptions[image_id].append(desc)
# 	return descriptions



# filename = 'D:/programs/barati/image captioning/flicker8k/dataset/Flickr8k_text/Flickr_8k.devImages.txt'
# train = load_set(filename)
# print('Dataset: %d' % len(train))

# # descriptions
# train_descriptions = load_clean_descriptions('descriptions_flicker8k.txt', train)
# print('Descriptions: train=%d' % len(train_descriptions))



# train2=list(train)
# filename = 'D:/programs/barati/image captioning/flicker8k/dataset/Flickr8k_Dataset/Flicker8k_Dataset/'

# img_name_val_orginal2=[]
# for i in range(len(train)):
#     img_name_val_orginal2.append(filename+train2[i]+'.jpg')



# from keras.preprocessing import image
# number_of_test= len(img_name_val_orginal2)#3000
# test_image = []
# #len(list_image)
# for i in range(number_of_test):
#     img = image.load_img(img_name_val_orginal2[i],target_size=(200,200,3))
#     img = image.img_to_array(img)
#     img = img/255
# #    image = preprocess_input(image)
#     test_image.append(img)
# # X_test = np.array(test_image)
# # del test_image


# num_of_concept=3
# weight_concept=[]
# index_concept_test=[]
# actual, predicted = list(), list()
# predicted2=list()

# #for k in range (len(X)):
# for k in range (len(test_image)):
#     if k % 100 == 0:
#       print(k)
#     img=test_image[k]#X_test[k]
# #    img=X[k]
#     top_3=[]
#     proba = model_concept_all.predict(img.reshape(1,200,200,3))
#     top_3 =((-proba).argsort())[0][:num_of_concept]
#     weight_concept.append(proba[0][top_3])
#     result=[]
#     for i in range(len(top_3)):
#         result.append(tokenizer_concept_all.index_word[round(top_3[i])])
#     #a=int(result)
#     index_concept_test.append(result)
    
#     p=[]
#     act=[]

#     for j in range(len(result)):
#         a=int(result[j])
        
#         image = img_name_val_orginal2[k]
#         id=image_id_val[k]
#         result2, attention_plot = evaluate(image,a)
#         r2=result2[0:(len(result2)-1)]
#         r3=' '.join(r2)
#         p.append(r2)
#     actual.append(train_descriptions[train2[k]])
#     predicted.append(p[0])#p[0]+p[1]+p[2]
#     predicted2.append(p)


# # actual2=[]
# # for i in range(len(actual)):
# #     act2=[]
# #     for j in range(3):
# #         act=[]
# #         for k in range(5):
# #             act.append(actual[i][j][k].split())
# #         # act2.append(act)
# #     actual2.append(act)

# # del actual
# # actual=actual2
# # del actual2, act, act2


# from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
# # calculate BLEU score
# print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
# print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
# print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)));
# print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)));














