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

#        #Draw circles
#        base_im = np.zeros((100,100))



        # Get equalized gray image
        print("Getting gray image")
        self.get_image_gray()

        # Extract labels and properties from the image
        print("Extracting labels and properties")
        self.extract_label_prop(sigma=1.7, nbr_ite_morph=4, minimum_area=400, savefig=False)

        # Extract arrow properties
        print("Extracting arrow properties")
        self.arrow_info = self.process_arrow(inv_angle=0.6, plotfig=False, savefig=False)

        # Get sub images
        print("Getting sub images")
        self.get_sub_images_and_blob(savefig=False)

        # Extract circle info
        print("Getting circle info")
        self.circle_info = self.extract_circle_info(min_size = 50, savefig=False, plotfig=False)

        # Get arrow index
        print("Getting arrow index")
        self.arrow_index = self.get_arrow_sub_image_index()


        if False:
            for index in range(len(self.sub_images)):
                for index2 in range(len(self.sub_images[index])):
                    if index2 == self.arrow_index[index][0] or index2 == self.circle_info[index][0]:
                        plt.imshow(self.sub_images[index][index2])
                        plt.show()

        # Image pairing
        print("Finding matched pairs")
        self.matched_pairs = self.image_pairing()

        # Classification
        print("Classifying")
        self.predictions = self.do_classification(self.sub_images, self.sub_images_blob)


        # Figure infos
        print("Getting figures info")
        self.figures_info = self.get_figures_info()

    def get_arrow_info(images):
        self.images = images
        self.get_image_gray()
        self.arrow_info = self.process_arrow(inv_angle=0.6, plotfig=False, savefig=False)
        return self.arrow_info



    def get_figures_info(self):
        figures_info = []
        for index in range(len(self.sub_images)):
            figures_info.append({})
            for index2 in range(len(self.sub_images[index])):
                # Skip arrow and circle
                if index2 == self.arrow_index[index][0] or index2 == self.circle_info[index][0]:
                    continue
                base_index = index2
                try:
                    matching_pair = self.matched_pair[index][index2]
                except:
                    matching_pair = -1
                bbox = self.region_props[index][index2].bbox
                center = ((bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2)
                number = self.predictions[index][index2]
                figures_info[index][index2] = (center, number, matching_pair, bbox)
        return figures_info

    def image_pairing(self):
        hu_distances = self.get_hu_distances(self.region_props)
        self.matched_pair = self.get_matching_pairs(hu_distances)
        return self.matched_pair

    def get_arrow_sub_image_index(self):
        arrow_index = []
        for index in range(len(self.sub_images)):
            arrow_index.append([])
            for index2, prop in enumerate(self.region_props[index]):
                arrow_center = self.arrow_info[index][0]
                bbox = prop.bbox
                if ((arrow_center[0] < bbox[2]) and (arrow_center[0] > bbox[0]) and
                    (arrow_center[1] < bbox[3]) and (arrow_center[1] > bbox[1])):
                    arrow_index[index].append(index2)
        return arrow_index

    def get_sub_images_and_blob(self, savefig=False):
        self.sub_images = self.get_sub_images(self.images_gray, self.region_props,
                                         self.label_im)

        self.sub_images_blob = self.get_sub_images_blob(self.sub_images)

        if savefig:
            for index in range(len(self.sub_images)):
                self.show_sub_images(self.sub_images, image_index=index)

        if savefig:
            for index in range(len(sub_images_blob)):
                self.show_sub_images(self.sub_images_blob, image_index=index, prefix="blob_")

    def extract_circle_info(self, min_size=100, savefig=False, plotfig=False):
        circle_index = self.find_circle_index(min_size = min_size, savefig=savefig, printfig=plotfig)
        circle_info = []
        for index in range(self.images_gray.shape[0]):
            prop = self.region_props[index][circle_index[index]]
            bbox = prop.bbox
            bbox_center = ((bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2)
            circle_info.append((circle_index[index], bbox_center))
        return circle_info

    def check_fourier_is_ok(self, fourier_descriptor):
        target_fourier = [(5300, 7000), (20, 300), (910, 1800)]
        fourier_ok = True
        for i in range(len(target_fourier)):
            if fourier_descriptor[i+1] < target_fourier[i][0]:
                fourier_ok = False
            elif fourier_descriptor[i+1] > target_fourier[i][1]:
                fourier_ok = False
        return fourier_ok

    def get_arrow_contours(self, plotfig=False):
        gradient_map = np.zeros(np.shape(self.images_gray))
        arrow_contours = {}
        for index in range(len(self.images_gray)):
            gradient_im = filters.sobel(self.images_gray[index])
            thresh = filters.threshold_otsu(gradient_im)
            gradient_im[gradient_im < thresh] = 0
            thresh = filters.threshold_otsu(gradient_im)
            gradient_im[gradient_im < thresh] = 0
            gradient_im[gradient_im >= thresh] = 1
            sub_image = ndi.binary_closing(gradient_im, iterations=1)
            contours = measure.find_contours(gradient_im, level=0.8)

            for i, contour in enumerate(contours):
                complex_contour = contour[:,0] + (contour[:,1] * 1j)
                fourier_transform = np.fft.fft(complex_contour)
                fourier_descriptor = np.absolute(fourier_transform)
                if self.check_fourier_is_ok(fourier_descriptor):
                    arrow_contours[index] = contour
                    if plotfig:
                        plt.plot(contour[:,1], contour[:,0],linewidth=2,
                                  label="{}".format(fourier_descriptor[1:4]))
            if plotfig:
                plt.imshow(gradient_im)
                plt.legend()
                plt.show()

        return arrow_contours

    def get_arrow_im(self, arrow_contours, savefig=False):
        arrows = np.zeros(self.images_gray.shape).astype(int)
        for key in arrow_contours.keys():
            contour = arrow_contours[key]
            rr, cc = skimage.draw.polygon(contour[:,0], contour[:,1],
                                          self.images_gray.shape[1:3])
            arrows[key, rr,cc] = 1

        if savefig:
            self.print_images(self.arrows, title="Arrows")
        return arrows

    def get_arrow_info(self, arrow_im, arrow_props, inv_angle = 0.6, plotfig=False):
        arrow_info = []
        for index in range(len(arrow_props)):
            prop = arrow_props[index][0]
            bbox = prop.bbox
            bbox_center = ((bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2)
            orientation = prop.orientation

            proj_x_neg = np.sum(arrow_im[index, :, bbox[1]:np.floor(bbox_center[1]).astype(int)])
            proj_x_pos = np.sum(arrow_im[index, :, np.ceil(bbox_center[1]).astype(int):bbox[3]])
            if plotfig:
                plt.imshow(arrow_im[index, :, bbox[1]:np.floor(bbox_center[1])])
                plt.show()
                plt.imshow(arrow_im[index, :, np.ceil(bbox_center[1]):bbox[3]])
                plt.show()

            if proj_x_neg > proj_x_pos:
                direction_x = -1
            else:
                direction_x = 1

            # If the angle is too big, the projection will give us the inverse result
            if np.absolute(prop.orientation) > inv_angle*(np.pi/2):
                direction_x = -direction_x
            arrow_info.append((bbox_center, orientation, direction_x))
        return arrow_info

    def process_arrow(self, inv_angle=0.6, plotfig=False, savefig=False):
        arrow_contours = self.get_arrow_contours()

        arrow_im = self.get_arrow_im(arrow_contours, savefig=savefig)

        arrow_props = self.get_region_props(arrow_im)

        arrow_info = self.get_arrow_info(arrow_im, arrow_props, inv_angle, plotfig)

        return arrow_info

    def get_image_gray(self):
        images_eq = self.equalize_images(self.images)
#        self.print_images(images_eq, title="Images Equalized")

        self.images_gray = skimage.color.rgb2gray(images_eq)
#        self.print_images(self.images_gray, axis=True, title="Images Grayscale")

    def extract_label_prop(self, sigma=1.7, nbr_ite_morph=4, minimum_area=400, savefig=False):
        images_edges = self.get_edges(self.images_gray, sigma=1.8)
#        self.print_images(images_edges, title="Images Edges")

        images_morph = self.morph_to_get_shape(images_edges, nbr_ite=4)
#        self.print_images(images_morph, title="Images Morphological")

        self.label_im = self.get_labels(images_morph)
#        self.print_images(self.label_im, title="Images label")

        self.region_props = self.get_region_props(self.label_im)

        # Remove small images
        self.label_im = self.remove_small_labels(self.label_im, self.region_props, minimum_area=400)
        # Recalculate region props
        self.region_props = self.get_region_props(self.label_im)

    def find_circle_index(self, min_size= 100, savefig=False, printfig=False):
        scaled_sub_images = self.scale_up_sub_images(self.sub_images_blob, self.region_props, min_size =min_size)
        #scaled_sub_images_blob = self.get_sub_images_blob(scaled_sub_images)
        scaled_sub_images_blob = []
        for index in range(len(scaled_sub_images)):
            scaled_sub_images_blob.append([])
            for index2 in range(len(scaled_sub_images[index])):
                local_im = scaled_sub_images[index][index2]
                thresh = filters.threshold_otsu(local_im)
                local_im[local_im < thresh] = 0
                thresh = filters.threshold_otsu(local_im)
                local_im[local_im < thresh] = 0
                local_im[local_im >= thresh] = 1
                scaled_sub_images_blob[index].append(local_im)

        SAVE_SCALED_SUBIMAGES = True
        if SAVE_SCALED_SUBIMAGES:
            for index in range(len(scaled_sub_images)):
                self.show_sub_images(scaled_sub_images, image_index=index, prefix="scaled_")

        SAVE_SCALED_SUBIMAGES_BLOB = True
        if SAVE_SCALED_SUBIMAGES_BLOB:
            for index in range(len(scaled_sub_images_blob)):
                self.show_sub_images(scaled_sub_images_blob, image_index=index, prefix="scaled_blob_")

        compacity = self.get_sub_compacity(scaled_sub_images_blob)

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

        return self.min_compacity_index


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

    def get_region_props(self, label_im):
        region_props = []
        for index in range(len(label_im)):
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
                if i == self.circle_info[index][0] or i == self.arrow_index[index][0]:
                    argmax_array[index].append(-1)
                    continue
                max_distance = 0
                for j in range(len(hu_distances[index][0])):
                    # don't calculate distance with yourself
                    if i == j:
                        continue
                    # skip circle and arrow
                    if j == self.circle_info[index][0] or j == self.arrow_index[index][0]:
                        continue

                    local_distance = hu_distances[index][i][j]
                    if local_distance > max_distance:
                        max_distance = local_distance
                        max_index = j
                argmax_array[index].append(max_index)

        matched_pair = []
        unmatched_val = []
        for index in range(len(hu_distances)):
            matched_pair.append({})
            unmatched_val = []
            good_match = 0
            bad_match = 0
            for i in range(len(hu_distances[index])):
                local_argmax = argmax_array[index][i]

                if local_argmax == -1:
                    continue

                pair_argmax = argmax_array[index][local_argmax]

                if pair_argmax == i:
                    self.print_local("Good match for {} with {}".format(i, local_argmax))
                    good_match += 1
                    matched_pair[index][i] = local_argmax
                else:
                    self.print_local("Bad match for {} with {}, whilst pair was {}".format(i, local_argmax, pair_argmax))
                    bad_match += 1
                    unmatched_val.append(i)

            new_unmatched_val = []
            if bad_match > 1:
                match_index = {}
                for i in unmatched_val:
                    max_dist = 0
                    for j in unmatched_val:
                        if i == j:
                            continue
                        local_distance = hu_distances[index][i][j]
                        if local_distance > max_dist:
                            max_dist = local_distance
                            max_index = j
                    match_index[i] = max_index
                for key in match_index.keys():
                    match = match_index[key]
                    other_match = match_index[match]
                    if other_match == key:
                        self.print_local("NEW Good match for {} with {}".format(key, match))
                        good_match += 1
                        bad_match -= 1
                        matched_pair[index][key] = match
                    else:
                        new_unmatched_val.append(key)

            # TODO: only match numbers with non-numbers

            # HACK
            if len(new_unmatched_val) == 2:
                key0 = new_unmatched_val[0]
                key1 = new_unmatched_val[1]
                matched_pair[index][key0] = key1
                matched_pair[index][key1] = key0





            self.print_local("{}: Good match: {}, Bad match: {}".format( \
                             index, good_match, bad_match)
                            )
            cumulative_good_match += good_match
            cumulative_bad_match += bad_match

        self.print_local("Cumulative: Good match: {}, Bad match: {}".format( \
                         cumulative_good_match, cumulative_bad_match)
                        )

        return matched_pair

    def extraction(self, image, blob):

        # Otsu's Thresholding
        thresh_otsu = filters.threshold_otsu(image)
        image_filt = (image > thresh_otsu)

        # Blob Extraction
        # Remove line below
        blob = ndi.binary_fill_holes(~image_filt)

        image_segm = blob & image_filt

        #number_segm = ndi.binary_closing(image_segm)
        number_segm = image_segm

        return number_segm

    def reshape_number(self, image) :

        properties = skimage.measure.regionprops(image.astype(int))

        min_row = -1
        max_row = -1
        min_col = -1
        max_col = -1

        for prop in properties :
            min_row = prop.bbox[0]
            min_col = prop.bbox[1]
            max_row = prop.bbox[2]
            max_col = prop.bbox[3]

        if (min_row != -1) & (min_col != -1) & (max_row != -1) & (max_col != -1) :

            extracted_number = image[min_row:max_row, min_col:max_col]

            maxi = np.max(np.shape(extracted_number)) + 8
            #mini = np.min(np.shape(extracted_number))
            #ind_mini = np.where(np.shape(extracted_number) == mini)

            offset_row = np.round((maxi-extracted_number.shape[0])/2).astype(int)
            offset_col = np.round((maxi-extracted_number.shape[1])/2).astype(int)


            extended_number = np.zeros([maxi, maxi])

            extended_number[offset_row:offset_row+extracted_number.shape[0],
                            offset_col:offset_col+extracted_number.shape[1]] = extracted_number

            #index_begin = np.round((maxi-mini)/2).astype(int)
            #index_finish = index_begin + mini

            #if ind_mini == 0:
            #    extended_number[4:maxi-4, index_begin:index_finish] = extracted_number
            #else:
            #    extended_number[index_begin:index_finish, 4:maxi-4] = extracted_number

            extended_number_resh = skimage.transform.resize(extended_number, [28, 28], mode='edge')

            return extended_number_resh
        else :
            return np.zeros((28,28))

    def predict_number(self, number, mlp) :
        # Reshape image
        extended_number = self.reshape_number(number)
        number_resh = extended_number.reshape((1, 28*28))#/255
        predicted_number = mlp.predict(number_resh)

        return predicted_number[0] #mlp.predict(number_resh)

    def do_classification(self, sub_images, sub_images_blob):
        # Load classifier and scaler
        with open('MLPClassifier.pickle', 'rb') as f :
            clf = pickle.load(f)

        predictions = []
        for index in range(len(sub_images)):
            predictions.append([])
            for index2 in range(len(sub_images[index])):
                image = sub_images[index][index2]
                blob = sub_images_blob[index][index2]


                number_segm = self.extraction(image, blob)

                percent_white = np.sum(number_segm)/number_segm.size

                if  (percent_white > 0.02) & (percent_white < 0.5) :
                    extended_number = self.reshape_number(number_segm)
                    predicted_number = self.predict_number(extended_number, clf)
                else :
                    predicted_number = -1 # No number in the image
                predictions[index].append(predicted_number)
        return predictions
