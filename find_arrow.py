import tarfile
import os
import pickle
from time import gmtime, strftime

import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

from skimage import exposure, feature, measure
import skimage.filters as sk_filters

import scipy
from scipy import ndimage as ndi
from scipy import spatial

import seaborn as sns


VERBOSE = True

RESULT_DIR = "result"
MAX_NBR_IMAGES = 23
WITH_TIMESTAMP = False

try:
    os.mkdir(RESULT_DIR)
except:
    pass

with open(os.path.join(RESULT_DIR, "images_only_shapes.pickle"), 'r') as f:
    images = pickle.load(f)

## Import Images
#file_names = []
#im_names = []
#data_path = 'Parcours'
#count = 0
#for file in os.listdir(data_path):
#    if file.endswith(".jpg"):
#        im_names.append(file.split('.')[0])
#        file_names.append(os.path.join(data_path,file))
#        count += 1
#        if count == MAX_NBR_IMAGES:
#            break
#
#ic = skimage.io.imread_collection(file_names)
#images = skimage.io.concatenate_images(ic)
#if VERBOSE:
#    print('Number of images: ', images.shape[0])
#    print('Image size: {}, {} '.format(images.shape[1], images.shape[2]))
#    print('Number of color channels: ', images.shape[-1])
im_names = [index for index in range(images.shape[0])]

def print_images(images, im_names=im_names, axis=False, size=(10,20), title=None, to_file=False):
    if not VERBOSE:
        return

    fig, axes = plt.subplots(images.shape[0]//2, 2, figsize=size)
    for ax, im, nm in zip(axes.ravel(), images, im_names):
        ax.imshow(im, cmap="gray")
        if not axis:
            ax.axis('off')
        ax.set_title(nm)
    if title:
        fig.suptitle(title)
    if to_file:
        timestamp = strftime("%m%d_%H_%M_%s", gmtime())
        if WITH_TIMESTAMP:
            image_title = "{}_{}.png".format(title, timestamp)
        else:
            image_title = "{}.png".format(title)
        plt.savefig(os.path.join(RESULT_DIR, image_title))
    else:
        plt.show()
    plt.close()

# Lets plot the distribution of values in the image
def show_distplot(array, show=True, nb_bins=256, ax=None,
                                logy=True, title=None):
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.hist(array.ravel(), bins=nb_bins, log=logy, label=title)
        if title:
            plt.title(title)
        if show:
            plt.show()

def equalize_images(images, color=True):
    images_eq = np.zeros(images.shape)
    for index in range(images.shape[0]):
        if color:
            for chroma in range(images.shape[3]):
                images_eq[index][:,:,chroma] = \
                    exposure.equalize_adapthist(images[index][:,:,chroma])
        else:
            images_eq[index] = exposure.equalize_adapthist(images[index])
    return images_eq

images_eq = equalize_images(images)
print_images(images_eq, title="Images Equalized", to_file=True)

images_gray =  (skimage.color.rgb2gray(images)*255).astype(np.uint8)
print_images(images_gray, axis=True, title="Images Grayscale", to_file=True)


dist_to_plot = [21, 12, 20, 13, 11, 14]
for index in dist_to_plot:#range(images_gray.shape[0]):
    pass
    #show_distplot(images_gray[index], title=str(index))

def get_bincount(images, max_val=255):
    # Note: will only work with grayscale images
    bincount={}
    for index in range(images.shape[0]):
        bincount[index] = np.bincount(np.reshape(images[index],
                                                 (images.shape[1]*images.shape[2])))
    return bincount

def get_local_extrema(bincount, order=2, extrema='min'):
    local_extrema = {}
    for key in bincount.keys():
        if extrema is 'min':
            local_extrema[key] = scipy.signal.argrelextrema(bincount[key], np.less, order=order)
        if extrema is 'max':
            local_extrema[key] = scipy.signal.argrelextrema(bincount[key], np.greater, order=order)
    return local_extrema

LOCAL_MIN_KERNEL_SIZE = 4
bincount = get_bincount(images_gray)
local_min = get_local_extrema(bincount, order = LOCAL_MIN_KERNEL_SIZE, extrema='min')
local_max = get_local_extrema(bincount, order = LOCAL_MIN_KERNEL_SIZE, extrema='max')

def threshold_image(image, threshold, value=255, min_val=0):
    thresholded_image = np.zeros(image.shape)
    mask_max = image < threshold
    mask_min = image > min_val
    mask = np.logical_and(mask_max, mask_min)
    thresholded_image[mask] = value
    return thresholded_image

images_black_pix = np.zeros(images_gray.shape)
for index in range(images_black_pix.shape[0]):
    images_black_pix[index] = threshold_image(images_gray[index], threshold=local_max[index][0][1])


print_images(images_black_pix)

def morph_to_get_shape(images):
    images_morph = np.zeros(images.shape)
    for index in range(images.shape[0]):
        images_morph[index] = images[index]
        # Fill the holes
        images_morph[index] = ndi.binary_fill_holes(images_morph[index])
        images_morph[index] = ndi.binary_erosion(images_morph[index], iterations = 3,
                                                 structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]]))
        images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = 3,
                                                 structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]]))
        # Erode
        #images_morph[index] = ndi.binary_erosion(images_morph[index], iterations = 3,
        #                                         structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]]))
        # Dilate
        #images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = 3,
        #                                         structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]]))
#        images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = 6)
    return images_morph

images_morph = morph_to_get_shape(images_black_pix)
print_images(images_morph, title="Images Morphological")#, to_file=True)

