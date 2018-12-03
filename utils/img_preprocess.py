import cv2
import numpy as np
from PIL import Image

#size = (299, 299) # (299,299) for inceptions, (224, 224) for others
def brightness_tuning(img, ceiling = 255):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if np.max(img[:, :, 2]) < ceiling:
        max_value = np.max(img[:, :, 2])
        ratio = 255 / max_value
        img[:, :, 2] = img[:, :, 2] * ratio
        img[:, :, 2] = np.where(img[:, :, 2] > 255, 255, img[:, :, 2])
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

def img_process_PIL(path, seq_1, transformations, transform, size):
    # random room and central crop
    img = cv2.imread(path) 
    b, g, r = cv2.split(img)
    img = cv2.merge((r, g, b))
    img = img_process_transform(img, seq_1, transform, size) 
    img = transformations(img)
    return img

def img_process_transform(img, augmentor, transform, size):
    img = np.expand_dims(img, axis = 0)
    if transform:
        img = augmentor.augment_images(img)
    img = np.squeeze(img, axis = 0)
    img = cv2.resize(img, size)
    img = Image.fromarray(img.astype('uint8'), mode="RGB")
    return img