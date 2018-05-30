#!/usr/bin/env python3
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

from scipy import ndimage as ndi
from scipy import spatial

import seaborn as sns

import argparse

# Define the parser
parser = argparse.ArgumentParser(description='Process image')
parser.add_argument('image',  help='The image file to process')
parser.add_argument('--result', '-r', help='Name of the result directory')
parser.add_argument('--prefix', '-p', help='Prefix to append to the results')
parser.add_argument('--to_file', '-f',  action='store_true', help='Save figures to file')
parser.add_argument('--verbose', '-v', action='store_true', help='Should we do verbose output')
parser.add_argument('--timestamp', '-t', action='store_true', help='Add timestamp to results')

#parser.parse_args(args=['image_name', '--result', '--prefix', '--to_file'], namespace=image)
args = parser.parse_args()

# Define the print function
if args.verbose:
    print_local = print
    VERBOSE = True
else:
    VERBOSE = False
    def print_local(*args):
       pass

IMAGE_PATH = args.image
IMAGE_BASE, IMAGE_NAME = os.path.split(IMAGE_PATH)
print_local(IMAGE_PATH, IMAGE_BASE, IMAGE_NAME)
print_local(args.result)
im_names = [IMAGE_NAME]

if args.result:
    RESULT_DIR = args.result
else:
    RESULT_DIR = "result"
try:
    os.mkdir(RESULT_DIR)
except:
    pass

if args.timestamp:
    WITH_TIMESTAMP = True
else:
    WITH_TIMESTAMP = False

if args.to_file:
    SAVE_TO_FILE = True
else:
    SAVE_TO_FILE = False

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

#TODO: NOTE: we do a bit of a hack with the images, this code can work with N images,
#            but the argument can only manage 1 image name
ic = skimage.io.imread_collection([IMAGE_PATH])
images = skimage.io.concatenate_images(ic)

print_local('Number of images: ', images.shape[0])
print_local('Image size: {}, {} '.format(images.shape[1], images.shape[2]))
print_local('Number of color channels: ', images.shape[-1])

def print_images(images, im_names=im_names, axis=False, size=(10,20), title=None, to_file=False):
    if not VERBOSE:
        return

    if images.shape[0] >= 2:
        im_per_row = 2
    else:
        im_per_row = 1
    fig, axes = plt.subplots(images.shape[0]//im_per_row, im_per_row, figsize=size)

    if images.shape[0] > 1:
        axes = axes.ravel()
    else:
        axes = [axes]

    for ax, im, nm in zip(axes, images, im_names):
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

#print_images(images, im_names)

# Equalize the images
def equalize_images(images, color=True):
    images_eq = np.zeros(images.shape)
    for index in range(images.shape[0]):
        if color:
            for chroma in range(images.shape[3]):
                images_eq[index][:,:,chroma] = \
                    exposure.equalize_hist(images[index][:,:,chroma])
        else:
            images_eq[index] = exposure.equalize_hist(images[index])
    return images_eq

images_eq = equalize_images(images)
print_images(images_eq, title="Images Equalized", to_file=SAVE_TO_FILE)

# Transform the images to grayscale
images_gray =  skimage.color.rgb2gray(images_eq)
print_images(images_gray, axis=True, title="Images Grayscale", to_file=SAVE_TO_FILE)

def get_edges(images, sigma=2.0):
    # Get the image edges
    images_edges = np.zeros(images.shape)
    for index in range(images.shape[0]):
        images_edges[index] = feature.canny(images[index], sigma=sigma)
    return images_edges

images_edges = get_edges(images_gray)
print_images(images_edges, title="Images Edges", to_file=SAVE_TO_FILE)

def morph_to_get_shape(images, nbr_ite=3):
    images_morph = np.zeros(images.shape)
    for index in range(images.shape[0]):
        images_morph[index] = images[index]
        # Dilate the image edges
        #images_morph[index] = ndi.binary_dilation(images[index])
        # Fill the holes
        images_morph[index] = ndi.binary_fill_holes(images_morph[index])
        # Erode
        for i in range(nbr_ite):
            if i % 2 is 0:
                images_morph[index] = ndi.binary_erosion(images_morph[index], iterations = 1,
                                                         structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]]))
            else:
                images_morph[index] = ndi.binary_erosion(images_morph[index], iterations = 1,
                                                         structure=np.asarray([[1,0,1],[0,1,0],[1,0,1]]))
        # Dilate
        for i in range(nbr_ite):
            if i % 2 is 0:
                images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = 1,
                                                         structure=np.asarray([[0,1,0],[1,1,1],[0,1,0]]))
            else:
                images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = 1,
                                                         structure=np.asarray([[1,0,1],[0,1,0],[1,0,1]]))
#        images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = 6)
    return images_morph

images_morph = morph_to_get_shape(images_edges)
print_images(images_morph, title="Images Morphological", to_file=SAVE_TO_FILE)

# get labels
shape_mask = images_morph == 1
label_im, nb_labels = ndi.label(shape_mask)
print_local(np.unique(label_im))
print_local(label_im.shape)

def get_images_hsv(images):
    """ Return "images" in the hsv format"""
    images_hsv = np.zeros(images.shape)
    for index in range(images.shape[0]):
        images_hsv[index] = skimage.color.rgb2hsv(images[index])
    return images_hsv

# Create hsv image representation
images_only_shape = np.copy(images)
#images_only_shape = get_images_hsv(images_only_shape)

# Remove the background from the images, i.e. keep only the shapes
for index in range(images_only_shape.shape[3]):
    temp_im = images_only_shape[:,:,:,index]
    temp_im[~shape_mask] = 0
    images_only_shape[:,:,:,index] = temp_im
print_images(images_only_shape, axis=True, title="Images Only_shapes", to_file=SAVE_TO_FILE)

with open(os.path.join(RESULT_DIR, "images_only_shapes.pickle"), 'wb') as f:
    pickle.dump(images_only_shape, f)


plt.cla()
region_props = measure.regionprops(label_im[0], images_gray[0])
NBR_OF_HU_MOMENTS = 7
weighted_hu_moments = np.zeros((len(region_props), NBR_OF_HU_MOMENTS))
for index, prop in enumerate(region_props):
    #print_local("********************************")
    #print_local("LABEL: {}".format(index))
    #print_local(prop.weighted_moments_hu)
    #print_local("centroid", prop.centroid)
    #print_local("convex area", prop.convex_area)
    #print_local("eccentricity", prop.eccentricity)
    #print_local("inertia tensor", prop.inertia_tensor)
    #print_local("major axis", prop.major_axis_length)
    #print_local("minor axis", prop.minor_axis_length)
    weighted_hu_moments[index] = prop.moments_hu
NBR_HU_KEPT=7
print_local(weighted_hu_moments[:,0:NBR_HU_KEPT])
hu_distances = spatial.distance_matrix(weighted_hu_moments[:,0:NBR_HU_KEPT], weighted_hu_moments[:,0:NBR_HU_KEPT], p=2)
for i in range(len(hu_distances)):
    for j in range(len(hu_distances[0])):
        if hu_distances[i][j] != 0.0:
            hu_distances[i,j] = 1/hu_distances[i,j]
sns.heatmap(hu_distances)
plt.savefig(os.path.join(RESULT_DIR, "hu_distances.png"))
plt.clf()

cmap = plt.cm.get_cmap('jet')
bounds = np.linspace(0, nb_labels+1, nb_labels+2)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
plt.imshow(label_im[0], cmap=cmap, norm=norm)
plt.colorbar()
plt.savefig(os.path.join(RESULT_DIR, "labels.png"))
#plt.show()
argmax_array = np.zeros(len(hu_distances))

for i in range(len(hu_distances)):
    argmax_array[i] =  int(np.argmax(hu_distances[i]))
    print_local("{}: {}".format(i, argmax_array[i]))

good_match = 0
for i in range(len(hu_distances)):
    if int(argmax_array[int(argmax_array[i])]) == int(i):
        print_local("Good match for {} with {}".format(i, argmax_array[i]))
        good_match += 1
    else:
        print_local("Bad match for {}".format(i))

print_local("Good match: {}, Bad match: {}".format(good_match, len(argmax_array)-good_match))


# TODO: Shapes are found pretty well
#       Need to separate rectangle with arrow from rest
#       Need to isolate circle if possible

# Find rectangle
