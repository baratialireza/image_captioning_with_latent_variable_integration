# -*- coding: utf-8 -*-

import tensorflow as tf
print(tf.__version__)

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
for val in annotations['annotations']:
  caption = f"<start> {val['caption']} <end>"
  image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
  image_path_to_caption[image_path].append(caption)


image_paths = list(image_path_to_caption.keys())
random.shuffle(image_paths)

# Select the first 6000 image_paths from the shuffled set.
# Approximately each image id has 5 captions associated with it, so that will
# lead to 30,000 examples.
train_image_paths = image_paths[:20000]
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

dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          map_func, [item1, item2], [tf.float32, tf.int64]),
          num_parallel_calls=tf.data.AUTOTUNE)

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

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

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

"""## Checkpoint"""
mydir='./checkpoints/train/'
for f in os.listdir(mydir):
    os.remove(os.path.join(mydir, f))

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

"""## Training

* You extract the features stored in the respective `.npy` files and then pass those features through the encoder.
* The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
* The decoder returns the predictions and the decoder hidden state.
* The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
* Use teacher forcing to decide the next input to the decoder.
* Teacher forcing is the technique where the target word is passed as the next input to the decoder.
* The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

"""

# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []

@tf.function
def train_step(img_tensor, target):
  loss = 0

  # initializing the hidden state for each batch
  # because the captions are not related from image to image
  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([word_to_index('<start>')] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          # using teacher forcing
          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss

EPOCHS = 101

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            average_batch_loss = batch_loss.numpy()/int(target.shape[1])
            print(f'Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}')
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    if epoch % 5 == 0:
        ckpt_manager.save()


    print(f'Epoch {epoch+1} Loss {total_loss/num_steps:.6f}')
    print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')




PATH2='D:/programs/barati/image captioning/mscoco/attention class vision/italy/result/loss plots'

plt.plot(loss_plot,label = ['global_net'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.legend(fontsize=5)
plt.savefig(PATH2+'/global_plot.png', dpi=300, bbox_inches='tight')
plt.show()


# Save a dictionary into a pickle file.
import pickle
pickle.dump(loss_plot, open(PATH2+"/loss_plot_global.p", "wb"))  # save it into a file named save.p
# -------------------------------------------------------------
# # Load the dictionary back from the pickle file.
# import pickle
# loss_plot = pickle.load(open(PATH2+"/loss_plot.p", "rb"))



"""## Caption!

* The evaluate function is similar to the training loop, except you don't use teacher forcing here. The input to the decoder at each time step is its previous predictions along with the hidden state and the encoder output.
* Stop predicting when the model predicts the end token.
* And store the attention weights for every time step.
"""

def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
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
result, attention_plot = evaluate(image)

print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(result))
plot_attention(image, result, attention_plot)

"""## Try it on your own images

For fun, below you're provided a method you can use to caption your own images with the model you've just trained. Keep in mind, it was trained on a relatively small amount of data, and your images may be different from the training data (so be prepared for weird results!)

"""

# image_url = 'https://tensorflow.org/images/surf.jpg'
# image_extension = image_url[-4:]
# image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)

# result, attention_plot = evaluate(image_path)
# print('Prediction Caption:', ' '.join(result))
# plot_attention(image_path, result, attention_plot)
# # opening the image
# Image.open(image_path)

"""# Next steps

Congrats! You've just trained an image captioning model with attention. Next, take a look at this example [Neural Machine Translation with Attention](https://www.tensorflow.org/text/tutorials/nmt_with_attention). It uses a similar architecture to translate between Spanish and English sentences. You can also experiment with training the code in this notebook on a different dataset.
"""

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


captions_val2014_tensorflowcore_results={}
captions_val2014_tensorflowcore2_results=[]
actual, predicted = list(), list()
print(len(img_name_val_orginal))
for i2 in range(len(img_name_val_orginal)) : #len(img_name_val_orginal)
  if i2 % 100 == 0:
    print(i2)
  c2={}
  # captions on the validation set
  rid = i2
  image = img_name_val_orginal[rid]
  id=image_id_val[rid]
  # real_caption = ' '.join([tf.compat.as_text(index_to_word(i).numpy()) for i in cap_val[rid] if i not in [0]])
  result, attention_plot = evaluate(image)
  # r1=real_caption.split()
  # actual.append(r1[1:(len(r1)-1)])
  actual.append(caption_val[image_id_val[rid]])
  r2=result[0:(len(result)-1)]
  r3=' '.join(r2)
  predicted.append(r2)
  c2['image_id']=id
  c2['caption']=r3
  captions_val2014_tensorflowcore_results[id]=r3
  captions_val2014_tensorflowcore2_results.append(c2)

actual2=[]
for i in range(len(actual)):
    act=[]
    for j in range(5):
        act.append(actual[i][j].split())
    actual2.append(act)

del actual
actual=actual2
del actual2

print(predicted[6])
print(image_id_val[6])
print(img_name_val_orginal[6])

print(captions_val2014_tensorflowcore_results[image_id_val[6]])
print(len(captions_val2014_tensorflowcore_results))

# import json
# PATH2='D:/programs/barati/image captioning/mscoco/attention class vision/italy/result'

# with open((PATH2+'/captions_val2014_tensorflowcore_results.json'), 'w') as f:
#     json.dump(captions_val2014_tensorflowcore_results, f)

# with open((PATH2+'/captions_val2014_tensorflowcore2_results.json'), 'w') as f:
#     json.dump(captions_val2014_tensorflowcore2_results, f)

from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
# calculate BLEU score
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)));
print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)));


# import pickle
# # save to file
# pickle.dump(actual,open(PATH2+'/actual_val.pkl', 'wb'))
# pickle.dump(predicted, open(PATH2+'/predicted_val.pkl', 'wb'))
# pickle.dump(img_name_val_orginal, open(PATH2+'/img_name_val.pkl', 'wb'))

# actual = pickle.load(open(PATH2+'/actual_val.pkl', 'rb'))
# predicted = pickle.load(open(PATH2+'/predicted_val.pkl', 'rb'))



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# annotation_file = 'D:/programs/barati/image captioning/mscoco/dataset/image_info_test2014/annotations/image_info_test2014.json'

# with open(annotation_file, 'r') as f:
#     annotations = json.load(f)

# image_id_val=[]
# img_name_val_orginal=[]
# for val in annotations['images']:
#   image_id_val.append(val['id'])
#   img_name_val_orginal.append(PATH+'/test2014/test2014/'+val['file_name'])
  

# print(len(image_id_val))
# key2=np.unique(image_id_val) 
# print(len(image_id_val))

# print(len(key2))
# print(key2[3])

# print(len(img_name_val_orginal))
# print(img_name_val_orginal[3])
# print(image_id_val[3])
# print(key2[3])

# captions_test2014_tensorflowcore_results={}
# captions_test2014_tensorflowcore2_results=[]
# actual, predicted = list(), list()
# for i2 in range(len(img_name_val_orginal)) : #len(img_name_val_orginal)
#   if i2 % 100 == 0:
#     print(i2)
#   c2={}
#   # captions on the validation set
#   rid = i2
#   image = img_name_val_orginal[rid]
#   id=image_id_val[rid]
#   result, attention_plot = evaluate(image)
#   r2=result[0:(len(result)-1)]
#   r3=' '.join(r2)
#   predicted.append(r2)
#   c2['image_id']=id
#   c2['caption']=r3
#   captions_test2014_tensorflowcore_results[id]=r3
#   captions_test2014_tensorflowcore2_results.append(c2)

# print(predicted[6])
# print(image_id_val[6])
# print(img_name_val_orginal[6])
# print(captions_test2014_tensorflowcore_results[image_id_val[6]])
# print(len(captions_val2014_tensorflowcore_results))

# # import json
# with open((PATH2+'/captions_test2014_tensorflowcore_results.json'), 'w') as f:
#     json.dump(captions_test2014_tensorflowcore_results, f)

# with open((PATH2+'/captions_test2014_tensorflowcore2_results.json'), 'w') as f:
#     json.dump(captions_test2014_tensorflowcore2_results, f)


# import pickle
# # save to file
# pickle.dump(predicted, open(PATH2+'/predicted_test.pkl', 'wb'))
# pickle.dump(img_name_val_orginal, open(PATH2+'/img_name_test.pkl', 'wb'))
# predicted = pickle.load(open(PATH2+'/predicted_test.pkl', 'rb'))



########################################################################################




# ################################# finding best blue score ####################################
res_all=[]
res_all2=[]
res_all2_ind=[]


res=[]
res2=[]
res2_ind=[]
for i in range(len(actual)):
    b=sentence_bleu(actual[i], predicted[i], weights=(1.0, 0, 0, 0))
    res.append(b)
    if b>.7:
        res2_ind.append(i)
        res2.append(b)
res_all.append(res)
res_all2.append(res2)
res_all2_ind.append(res2_ind)
    

np.mean(res_all)


# ############################# attention plot and show on best blue score #############################
predicted_best_blue_global=[]
attention_plot_best_blue_global=[]

for ws in range(len(res_all2_ind[0])):
    cd=[]
    rd=[]
    ap=[]
    rid = res_all2_ind[0][ws]
        
    image = img_name_val_orginal[rid]
    id=image_id_val[rid]
    result2, attention_plot = evaluate(image)
    
    rd.append(result2)
    ap.append(attention_plot)
predicted_best_blue_global.append(rd)
attention_plot_best_blue_global.append(ap)



rs= np.random.randint(0, len(res_all2_ind[0]))
rid = res_all2_ind[0][rs]
real_caption = caption_val[image_id_val[rid]] 
print('Real Caption:', real_caption)
print('Prediction Caption:', ' '.join(predicted_best_blue_global[rs]))
image = img_name_val_orginal[rid]
plot_attention(image, predicted_best_blue_global[rs], attention_plot_best_blue_global[rs])

from PIL import Image
image = Image.open(image)
image.show()

################################ save ################################
PATH2='D:/programs/barati/image captioning/mscoco/attention class vision/italy/result/loss plots'
# Save a dictionary into a pickle file.
import pickle
pickle.dump(predicted_best_blue_global, open(PATH2+"/predicted_best_blue_global.p", "wb"))
pickle.dump(attention_plot_best_blue_global, open(PATH2+"/attention_plot_best_blue_global.p", "wb"))

# -------------------------------------------------------------
# # Load the dictionary back from the pickle file.
# import pickle
# predicted_best_blue = pickle.load(open(PATH2+"/predicted_best_blue.p", "rb"))
# attention_plot_best_blue = pickle.load(open(PATH2+"/attention_plot_best_blue.p", "rb"))
# concept_detected = pickle.load(open(PATH2+"/concept_detected.p", "rb"))



# ################################
# random with change caption availibility
# ################################
rs= np.random.randint(0, len(res_all2_ind[0]))
rid = res_all2_ind[0][rs]
real_caption = caption_val[image_id_val[rid]] 
print('Real Caption:', real_caption)
image = img_name_val_orginal[rid]
result2, attention_plot = evaluate(image)
print('Prediction Caption:', ' '.join(result2))
plot_attention(image, result2, attention_plot)



 
from PIL import Image
image = Image.open(image)
image.show()




