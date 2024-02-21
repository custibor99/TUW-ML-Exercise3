import time
from datetime import datetime
from matplotlib import pyplot as plt
from skimage.color import rgb2lab
from skimage.io import imread
import numpy as np
from skimage.color import lab2rgb
import tensorflow as tf

def log(text:str):
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f"[{dt_string}]{text}")

def custom_logger(func):
    def wrapper(*args, **kwargs):
        start = time.time_ns()
        result = func(*args, **kwargs)
        end = time.time_ns()
        dur = round((end - start) / 10**6,2)
        text = f"Function: {func.__name__}, Duration: {dur}"
        log(text)
        return result
    return wrapper

@custom_logger
def draw_lab(l:tf.Tensor, ab:tf.Tensor):
  img = tf.concat([l,ab], axis=2)
  img = (img + 1) / 2
  plt.imshow(img.numpy())
  plt.show()

@custom_logger
def draw_lab_channels(l:tf.Tensor, ab:tf.Tensor):
  img = tf.concat([l,ab], axis=2)
  img = (img + 1) / 2
  fig, ax = plt.subplots(1,3, figsize=(15,5))
  ax[0].set_title("Lightness channel")
  ax[1].set_title("A channel")
  ax[2].set_title("B channel")
  ax[0].imshow(img[:,:,0], cmap="gray")
  ax[1].imshow(img[:,:,1], cmap="gray")
  ax[2].imshow(img[:,:,2], cmap="gray")
  plt.show()

@custom_logger
def convert_lab_to_rgb(l:tf.Tensor, ab:tf.Tensor):
  l = (l + 1) * 50
  ab = ab * 128
  img = tf.concat([l,ab], axis=2)
  img = lab2rgb(img)
  return img



@tf.py_function(Tout=(np.uint8))
def load_image(filename:tf.Tensor) -> np.array:
    filepath = filename.numpy().decode("utf-8")
    img = imread(filepath)
    return img

@tf.py_function(Tout=(tf.float32, tf.float32))
def transform_image(img: np.array):
    try:
        img = rgb2lab(img).astype("float32")
    except:
        raise ValueError
    img = tf.convert_to_tensor(img)
    L = img[:,:,0] / 50. - 1.
    ab = img[:,:,1:] / 128.
    L = tf.reshape(L, [256,256,1])
    return L, ab

@custom_logger
def get_dataset(filepath:str, inMeamory = True) -> tf.data.Dataset:
  dataset = tf.data.Dataset.list_files(filepath)
  dataset = dataset.map(load_image)
  if inMeamory:
    dataset = list(dataset)
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
  dataset = dataset.map(transform_image)
  dataset = dataset.map(lambda x, y : (tf.ensure_shape(x, (256,256,1)), tf.ensure_shape(y, (256,256,2))) )
  return dataset