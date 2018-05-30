import os
import pickle
from time import gmtime, strftime

import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np

from skimage import exposure, feature, measure, filters

from scipy import ndimage as ndi
from scipy import spatial

import seaborn as sns


class ImageAnalysis:

    def __init__(self, images, im_names, result_dir="result", print_local=print, save_to_file=True, with_timestamp=False):
        self.images = images
        self.im_names = im_names
        self.result_dir = result_dir
        self.print_local = print_local
        self.to_file = save_to_file
        self.with_timestamp = with_timestamp

    def image_analysis(self):
        """Function which computes every steps of the image analysis"""

        images_eq = self.equalize_images(self.images)
#        self.print_images(images_eq, title="Images Equalized")

        self.images_gray = skimage.color.rgb2gray(images_eq)
#        self.print_images(self.images_gray, axis=True, title="Images Grayscale")

        images_edges = self.get_edges(self.images_gray, sigma=1.8)
#        self.print_images(images_edges, title="Images Edges")

        images_morph = self.morph_to_get_shape(images_edges, nbr_ite=4)
#        self.print_images(images_morph, title="Images Morphological")

        self.label_im = self.get_labels(images_morph)
#        self.print_images(self.label_im, title="Images label")

        self.region_props = self.get_region_props(self.images_gray, self.label_im)

        background_mask = images_morph == 0
        images_only_shape = self.remove_images_background(self.images, background_mask)
#        self.print_images(images_only_shape, axis=True, title="Images Only shapes")

        # gaussian, otsu, binary fill_holes
        # Remove small images
        self.label_im = self.remove_small_labels(self.label_im, self.region_props, minimum_area=400)
        # Recalculate region props
        self.region_props = self.get_region_props(self.images_gray, self.label_im)

        sub_images = self.get_sub_images(self.images_gray, self.region_props,
                                         self.label_im)

        sub_images_blob = self.get_sub_images_blob(sub_images)

        #max_bboxes = self.get_biggest_bbox_area(self.label_im, self.region_props)
        #for index in range(len(sub_images)):
        #    sub_figure_index = max_bboxes[index][0]
        #    plt.imshow(sub_images[index][sub_figure_index])
        #    plt.show()

        SAVE_SUBIMAGES = False
        if SAVE_SUBIMAGES:
            for index in range(len(sub_images)):
                self.show_sub_images(sub_images, image_index=index)

        SAVE_SUBIMAGES_BLOB = False
        if SAVE_SUBIMAGES_BLOB:
            for index in range(len(sub_images_blob)):
                self.show_sub_images(sub_images_blob, image_index=index, prefix="blob_")

        scaled_sub_images = self.scale_up_sub_images(sub_images_blob, self.region_props)
        scaled_sub_images_blob = self.get_sub_images_blob(scaled_sub_images)

        SAVE_SCALED_SUBIMAGES = True
        if SAVE_SCALED_SUBIMAGES:
            for index in range(len(scaled_sub_images)):
                self.show_sub_images(scaled_sub_images, image_index=index, prefix="scaled_")

        SAVE_SCALED_SUBIMAGES_BLOB = True
        if SAVE_SCALED_SUBIMAGES_BLOB:
            for index in range(len(scaled_sub_images_blob)):
                self.show_sub_images(scaled_sub_images_blob, image_index=index, prefix="scaled_blob_")

        # TODO: find out which image is the robot
        compacity = self.get_sub_compacity(scaled_sub_images_blob)
        #for index in range(len(compacity)):
        #    print("{}:".format(index), end='')
        #    for index2, item in enumerate(compacity[index]):
        #        print("{}_{}, ".format(index2, item), end='')
        #    print("")

        self.min_compacity_index = self.get_min_compacity_index(compacity)

        PRINT_MIN_COMPACITY_IMAGE = False
        if PRINT_MIN_COMPACITY_IMAGE:
            for index in range(len(compacity)):
                sorted_compacity = np.sort(compacity[index])
                for item in range(3):
                    mask = compacity[index] == sorted_compacity[item]
                    index2 = np.argmax(mask)
                    plt.imshow(scaled_sub_images_blob[index][index2])
                    plt.title("{}.{}: {}".format(index, index2, compacity[index][index2]))
                    plt.show()

        # TODO: remove circle and robot from the hu_distances
        #hu_distances = self.get_hu_distances(self.region_props)

        #self.matched_pair, self.unmatched_val = self.get_matching_pairs(hu_distances)

    def extract_label_prop(self, images, sigma=1.7, nbr_morph=4)


    def print_images(self, images, im_names=None, axis=False, size=(10,20), title=None, print_type="imshow", logy=True):
        #if not VERBOSE:
        #    return
        if not im_names:
            im_names = self.im_names

        if np.shape(images)[0] >= 2:
            im_per_row = 2
        else:
            im_per_row = 1
        fig, axes = plt.subplots(images.shape[0]//im_per_row, im_per_row, figsize=size)

        if np.shape(images)[0] > 1:
            axes = axes.ravel()
        else:
            axes = [axes]

        for ax, im, nm in zip(axes, images, im_names):
            if print_type == "imshow":
                ax.imshow(im, cmap="gray")
            elif print_type == "dist":
                ax.hist(im.ravel(), bins=256, log=logy)
            if not axis:
                ax.axis('off')
            ax.set_title(nm)
        if title:
            fig.suptitle(title)
        if self.to_file:
            timestamp = strftime("%m%d_%H_%M_%s", gmtime())
            if self.with_timestamp:
                image_title = "{}_{}.png".format(title, timestamp)
            else:
                image_title = "{}.png".format(title)
            plt.savefig(os.path.join(self.result_dir, image_title))
        else:
            plt.show()
        plt.close()

    def equalize_images(self, images, color=True):
        images_eq = np.zeros(images.shape)
        for index in range(images.shape[0]):
            if color:
                for chroma in range(images.shape[3]):
                    images_eq[index][:,:,chroma] = \
                        exposure.equalize_hist(images[index][:,:,chroma])
            else:
                images_eq[index] = exposure.equalize_hist(images[index])
        return images_eq


    def get_edges(self, images, sigma=2.0):
        # Get the image edges
        images_edges = np.zeros(images.shape)
        for index in range(images.shape[0]):
            images_edges[index] = feature.canny(images[index], sigma=sigma)
        return images_edges

    def morph_to_get_shape(self, images, nbr_ite=3):
        square_struct = np.asarray([[1,1],[1,1]])
        cross_struct = np.asarray([[0,1,0],[1,1,1],[0,1,0]])
        x_struct = np.asarray([[1,0,1],[0,1,0],[1,0,1]])

        images_morph = np.zeros(images.shape)
        for index in range(images.shape[0]):
            images_morph[index] = images[index]
            # Dilate the image edges
            images_morph[index] = ndi.binary_dilation(images[index], structure=cross_struct)
            # Fill the holes
            images_morph[index] = ndi.binary_fill_holes(images_morph[index])
            # Open
            images_morph[index] = ndi.binary_opening(images_morph[index], iterations = nbr_ite,
                                                     structure=cross_struct)
            #images_morph[index] = ndi.binary_erosion(images_morph[index], iterations = nbr_ite,
            #                                         structure=cross_struct)
            #images_morph[index] = ndi.binary_dilation(images_morph[index], iterations = nbr_ite,
            #                                         structure=cross_struct)
        return images_morph

    def get_labels(self, images):
        labels_im = np.zeros(images.shape)
        for index in range(images.shape[0]):
            shape_mask = images[index] == 1
            labels_im[index][shape_mask] = 1
            label_im, nb_labels = ndi.label(labels_im[index])
            labels_im[index] = label_im
        return labels_im.astype(int)


    def get_images_hsv(self, images):
        """ Return images in the hsv format"""
        images_hsv = np.zeros(images.shape)
        for index in range(images.shape[0]):
            images_hsv[index] = skimage.color.rgb2hsv(images[index])
        return images_hsv

    def remove_images_background(self, images, mask):
        images_only_shape = np.copy(images)
        images_only_shape[mask] = 0
        return images_only_shape

    def pickle_images(self, images, output_file="image.pickle"):
        with open(os.path.join(RESULT_DIR, output_file), 'wb') as f:
            pickle.dump(images_only_shape, f)

    def get_region_props(self, images, label_im):
        region_props = []
        for index in range(images.shape[0]):
            prop = measure.regionprops(label_im[index])
            region_props.append(prop)
        return region_props

    def get_biggest_bbox_area(self, label_im, region_props):
        max_bbox_area = []
        for index in range(len(label_im)):
            max_area = 0
            for index2, prop in enumerate(region_props[index]):
                bbox = prop.bbox
                bbox_area = np.absolute(bbox[3]-bbox[1] * np.absolute(bbox[2]-bbox[0]))
                if bbox_area > max_area:
                    max_area = bbox_area
                    max_bbox = bbox
                    max_area_index = index2
            max_bbox_area.append((max_area_index, max_bbox))
        return max_bbox_area


    def remove_small_labels(self, label_im, region_props, minimum_area=100):

        for index in range(len(label_im)):
            for prop in region_props[index]:
                bbox = prop.bbox
                bbox_area = np.absolute(bbox[3]-bbox[1]) * np.absolute(bbox[2]-bbox[0])
                if bbox_area < minimum_area:
                    # Remove all labels in the bbox
                    label_im[index, bbox[0]:bbox[2], bbox[1]:bbox[3]] = 0
        return label_im


    def get_sub_images(self, images, region_props, label_im, margin=5):

        def get_min_margin(val, margin):
            min_val = val-margin
            if min_val<0:
                min_val = 0
            return min_val

        def get_max_margin(val, margin, side):
            max_val = val+margin
            if max_val > side:
                max_val = side
            return max_val

        sub_images = []
        for index in range(images.shape[0]):
            local_sub = []
            for index2, prop in enumerate(region_props[index]):
                bbox = prop.bbox
                bbox_area = np.absolute(bbox[3]-bbox[1]) * np.absolute(bbox[2]-bbox[0])
                min_row = get_min_margin(bbox[0], margin)
                min_col = get_min_margin(bbox[1], margin)
                max_row = get_max_margin(bbox[2], margin, images.shape[1])
                max_col = get_max_margin(bbox[3], margin, images.shape[2])
                local_sub.append(images[index, min_row:max_row, min_col:max_col])

            sub_images.append(local_sub)

        return sub_images


    def show_sub_images(self, sub_images, image_index=0, size=(10,20), prefix=None):
        nb_label = len(np.unique(self.label_im[image_index]))
        cmap = plt.cm.get_cmap('jet')
        bounds = np.linspace(0, nb_label+1, nb_label+2)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(self.label_im[image_index], cmap=cmap, norm=norm)
        plt.colorbar()
        plt.title("Labels {}".format(self.im_names[image_index]))
        plt.savefig(os.path.join(self.result_dir, "labels_{}.png".format(image_index)))
        plt.close()

        nbr_sub_images = len(sub_images[image_index])
        if nbr_sub_images >= 2:
            im_per_row = 2
        else:
            im_per_row = 1
        nbr_row = np.ceil(nbr_sub_images/im_per_row).astype(int)
        fig, axes = plt.subplots(nbr_row, im_per_row, figsize=size)

        if nbr_sub_images > 1:
            axes = axes.ravel()
        else:
            axes = [axes]

        for index, ax in enumerate(axes):
            if index < nbr_sub_images:
                ax.imshow(sub_images[image_index][index], cmap="gray")
        fig.suptitle("Sub images {}".format(self.im_names[image_index]))
        plt.savefig(os.path.join(self.result_dir, "{}sub_images_{}.png".format(prefix, image_index)))
        plt.close(fig)

    def get_sub_images_blob(self, sub_images, sigma=1.4, min_density=.2, max_ite=20):
        sub_images_blob = []
        for index in range(len(sub_images)):
            sub_images_blob.append([])
            for index2, im in enumerate(sub_images[index]):
                image_density = 0
                close_ite = 0
                im_edges = feature.canny(im, sigma=sigma)
                # Make sure that a predefined fraction of the image is white pixels
                while image_density < min_density and close_ite < max_ite:
                    if close_ite > 0:
                        im_morph = ndi.binary_closing(im_edges, iterations = close_ite)
                    else:
                        im_morph = im_edges
                    im_blob = ndi.binary_fill_holes(im_morph)
                    open_ite = 0
                    nbr_of_labels = 100
                    # Make sure that we keep only 1 blob
                    while(nbr_of_labels > 1 and open_ite < max_ite):
                        if open_ite > 0:
                            im_morph2 = ndi.binary_opening(im_blob, iterations = open_ite)
                        else:
                            im_morph2 = im_blob
                        labels, nbr_of_labels = ndi.label(im_morph2)
                        open_ite += 1
                    image_density = np.count_nonzero(im_morph2)/im_morph2.size
                    close_ite +=1
 #                   plt.imshow(im_morph2)
 #                   plt.title("{}.{}: o{} c{}, d{}".format(index,index2, open_ite, close_ite, image_density))
  #                  plt.show()
                sub_images_blob[index].append(im_morph2)
        return sub_images_blob

    def scale_up_sub_images(self, sub_images, region_props, min_size = 100):
        new_sub_images = []
        for index in range(len(sub_images)):
            new_sub_images.append([])
            for index2 in range(len(sub_images[index])):
                bbox = region_props[index][index2].bbox
                row_len = bbox[2]-bbox[0]
                col_len = bbox[3]-bbox[1]
                if row_len < col_len:
                    min_len = row_len
                else:
                    min_len = col_len
                scale = min_size/min_len
                if scale > 1:
                    local_sub = skimage.transform.rescale(sub_images[index][index2], scale, mode="reflect")
                else:
                    local_sub = sub_images[index][index2]
                new_sub_images[index].append(local_sub)
        return new_sub_images

    def get_contours(self, sub_images):
        """Function to return the contours of binary images
        """
        contours = []
        for index in range(len(sub_images)):
            contours.append([])
            for index2 in range(len(sub_images[index])):
                contour = measure.find_contours(sub_images[index][index2], level=0.8)
                contours[index].append(contour)
        return contours

    def calculate_contours_fft(self, contours):
        """Function to calculate the fourier descriptors of contours
           For each contour:
              It calculates the array of complex values (x_n + j*y_n)
              It calculates the fast fourier transform of the complex values
              It takes the norm of the values
              It assign the norm in the "fourier_descriptors" dict that is returned
           @param contours a dictionnary of contours
        """
        fourier_descriptors = []
        # Create the complex number representation of the x and y position
        # given by the contour
        for index in range(len(contours)):
            fourier_descriptors.append([])
            for index2 in range(len(contours[index])):
                contour = contours[index][index2][0] # We only take the first contour
                complex_contour = contour[:,0] + (contour[:,1] * 1j)
                fourier_transform = np.fft.fft(complex_contour)
                fourier_descriptors[index].append(np.absolute(fourier_transform))
        return fourier_descriptors

    def get_sub_compacity(self, sub_images):
        compacities = []
        for index in range(len(sub_images)):
            compacities.append([])
            for index2, im in enumerate(sub_images[index]):
                #contour = measure.find_contours(im, level=0.8)
                #print(contour)
                #exit(0)
                perimeter = measure.perimeter(im, neighbourhood=4)
                area = np.count_nonzero(im)
                compacity = perimeter**2 / area
                if compacity < 4*np.pi:
                    print("Compacity is too small {} < {}".format(compacity, 4*np.pi))
                    compacity=100
                #print("{}.{}:".format(index, index2), perimeter, area, compacity)
                compacities[index].append(compacity)
        return compacities

    def get_min_compacity_index(self, compacity):
        min_compacity = []
        for index in range(len(compacity)):
            min_compacity.append(np.argmin(compacity[index]))
        return min_compacity


    def get_hu_distances(self, region_props):
        NBR_OF_HU_MOMENTS = 7
        #weighted_hu_moments = np.zeros((len(region_props[0]),
        #                                len(region_props[0,0]),
        #                                NBR_OF_HU_MOMENTS)
        #                              )
        #hu_distances = np.zeros((len(region_props[0]),
        #                         len(region_props[0][0]),
        #                         len(region_props[0][0]))
        #                        )
        hu_distances = []
        for index in range(len(region_props)):
            weighted_hu_moments = np.zeros((len(region_props[index]),
                                            NBR_OF_HU_MOMENTS)
                                          )

            for index2, prop in enumerate(region_props[index]):
                weighted_hu_moments[index2] = prop.moments_hu

            hu_distances.append(spatial.distance_matrix(weighted_hu_moments,
                                                        weighted_hu_moments,
                                                        p=2
                                                       )
                               )

            for i in range(len(hu_distances[index])):
                for j in range(len(hu_distances[index][0])):
                    if hu_distances[index][i,j] != 0.0:
                        hu_distances[index][i,j] = 1/hu_distances[index][i,j]

        return hu_distances

    def get_matching_pairs(self, hu_distances):
        cumulative_good_match=0
        cumulative_bad_match=0

        argmax_array = []
        for index in range(len(hu_distances)):
            argmax_array.append([])
            for i in range(len(hu_distances[index])):
                argmax_array[index].append(int(np.argmax(hu_distances[index][i])))
                #self.print_local("{}.{}: {}".format(index,i, argmax_array[index][i]))

        matched_pair = []
        unmatched_val = []
        for index in range(len(hu_distances)):
            matched_pair.append([])
            unmatched_val.append([])
            good_match = 0
            for i in range(len(hu_distances[index])):
                if int(argmax_array[index][int(argmax_array[index][i])]) == int(i):
                    #self.print_local("Good match for {} with {}".format(i, argmax_array[index][i]))
                    good_match += 1
                    matched_pair[index].append((i, argmax_array[index][i]))
                else:
                    #self.print_local("Bad match for {}".format(i))
                    unmatched_val[index].append(i)

            bad_match = len(argmax_array[index])-good_match
            self.print_local("{}: Good match: {}, Bad match: {}".format( \
                             index, good_match, bad_match)
                            )
            cumulative_good_match += good_match
            cumulative_bad_match += bad_match

        self.print_local("Cumulative: Good match: {}, Bad match: {}".format( \
                         cumulative_good_match, cumulative_bad_match)
                        )

        return matched_pair, unmatched_val

