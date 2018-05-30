#!/usr/bin/env python3

import os
import argparse

import skimage.io

from image_analysis import ImageAnalysis

def parser():
    # Define the parser
    parser = argparse.ArgumentParser(description='Process image')
    parser.add_argument('images',  nargs='+', help='The image file(s) to process')
    parser.add_argument('--result', '-r', help='Name of the result directory')
    parser.add_argument('--prefix', '-p', help='Prefix to append to the results')
    parser.add_argument('--to_file', '-f',  action='store_true', help='Save figures to file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Should we do verbose output')
    parser.add_argument('--timestamp', '-t', action='store_true', help='Add timestamp to results')

    #parser.parse_args(args=['image_name', '--result', '--prefix', '--to_file'], namespace=image)
    args = parser.parse_args()
    return args

def main(args):
    # Define the print function
    if args.verbose:
        print_local = print
        VERBOSE = True
    else:
        VERBOSE = False
        def print_local(*args):
           pass

    IMAGE_PATHS = args.images
    NBR_IMAGE = len(IMAGE_PATHS)

    im_names = []
    im_bases = []
    for path in IMAGE_PATHS:
        local_base, local_name = os.path.split(path)
        im_names.append(local_name)
        im_bases.append(local_base)

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

    ic = skimage.io.imread_collection(IMAGE_PATHS)
    images = skimage.io.concatenate_images(ic)

    print_local('Number of images: ', images.shape[0])
    print_local('Image size: {}, {} '.format(images.shape[1], images.shape[2]))
    print_local('Number of color channels: ', images.shape[-1])

    imageAnalysis = ImageAnalysis(images, im_names, result_dir=RESULT_DIR, print_local=print_local,
                  save_to_file=SAVE_TO_FILE)

    imageAnalysis.image_analysis()


if __name__ == "__main__":
    args = parser()
    main(args)
