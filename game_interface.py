#!/usr/bin/env python3

import os
import argparse

import skimage.io
import matplotlib.pyplot as plt
import numpy as np

from image_analysis import ImageAnalysis
from robot_control import RobotController
from iapr.webcam import WebcamVideoStream


# Tool config
print_local = print
RESULT_DIR = "result"
SAVE_TO_FILE = True


# Typed variables
imageAnalysis = None
wvs = WebcamVideoStream(src=0)
robotController = None


# PARAMETERS
CORR_FACTORS = False
NUMBER_OF_FIG = 10
DIST_MIN = 20


image_prefix = "test1"

def main():

    #TODO detect the count of forms
    image_counter = 1

    # 1) Create a webcam video stream
    #   Note: src=0 links to the USB Webcam of the computers provided for the project
    webcam = True
    if webcam:
        #wvs =

        images = get_images(image_counter)
        im_names = ["Analysis image"]
        for i in range(1000):
            get_images(i)
    else:
        ic =  skimage.io.imread_collection("Parcours/2.jpg")
        images = skimage.io.concatenate_images(ic)
        im_names = ["Test image"]

    # 2) first image analysis
    imageAnalysis = ImageAnalysis(images, im_names, result_dir=RESULT_DIR,
                                  print_local=print_local, save_to_file=SAVE_TO_FILE)

    imageAnalysis.image_analysis()
    figures_info = imageAnalysis.figures_info[0] # (center, number, matching_pair, bbox)
    circle_info = imageAnalysis.circle_info[0] # (index, bbox_center)
    arrow_info = imageAnalysis.arrow_info[0] # (bbox_center, orientation, direction_x)

    # 3) get target list and store it
    target_list = get_target_list(figures_info, circle_info)

    if len(target_list) < NUMBER_OF_FIG:
        print("WARNING: not enough figures have been detected")

    plt.imshow(images[0])
    for index in range(len(target_list)-1):
        x0 = target_list[index][1][0]
        x1 = target_list[index+1][1][0]
        y0 = target_list[index][1][1]
        y1 = target_list[index+1][1][1]

        plt.plot([y0,y1], [x0, x1], 'ro-')
    plt.savefig("gameplan.png")
    plt.close()

    # 4) calibrate robot
    robot_info_cal = get_robot_info()
    robotController = RobotController()
    robotController.calibration()
    robot_info = get_robot_info()
    robotController.calcFactorCalibr(robot_info_cal[0],
                                     robot_info_cal[1],
                                     robot_info_cal[2],
                                     robot_info[0],
                                     robot_info[1],
                                     robot_info[2]
                                    ) # (x_i, y_i, theta_i, x_f, y_f, theta_f)

    # Target are stored as : (key, center, is_number)

    # 5) Execute task
    for index, target in enumerate(target_list):
        target_point = target[1]
        x_t = target_point[1]
        y_t = target_point[0]
        is_number = target_point[2]
        if target_point[0] is "circle":
            finish = True

        on_shape = False
        while not on_shape:
            robotController.GoTo(robot_info[0],
                                 robot_info[1],
                                 robot_info[2],
                                 x_t, y_t
                                )
            robot_info = get_robot_info()
            if CORR_FACTORS:
                robotController.corrFactors( robot_info[0],
                                             robot_info[1],
                                             robot_info[2]
                                           )
            on_shape = robotController.check_on_the_shape(robot_info[0],
                                                          robot_info[1],
                                                          is_number,
                                                          finish = finish,
                                                          dist_min = DIST_MIN
                                                         )

# (bbox_center, orientation, direction_x)
def get_robot_info():
    arrow_info = get_arrow_info()
    orientation = arrow_info[1]
    direction_x = arrow_info[2]
    x = arrow_info[0][1]
    y = arrow_info[0][0]
    theta = robotController.convAngle(orientation, direction_x)
    return (x, y, theta)


def get_arrow_info():
    images = get_images(image_counter)
    arrow_info = imageAnalysis.get_arrow_info(images)
    return arrow_info

def get_images(image_counter):
    # Read most recent frame
    frame = wvs.read()

    images = np.zeros((1, frame.shape[0], frame.shape[1], frame.shape[2]))
    images[0] = frame
    plt.imshow(frame)
    plt.savefig("{}/{}_{}.png".format(RESULT_DIR, image_prefix, image_counter))
    image_counter += 1
    plt.close()
    return images

# Target are stored as : (key, center, is_number)
def get_target_list(figures_info, circle_info):
    target = []
    numbers = []
    for key in figures_info.keys():
        nbr = figures_info[key][1]
        if  nbr == -1:
            continue
        numbers.append((key, nbr))
    numbers.sort(key=lambda tup: tup[1])
    for number in numbers:
        # Append number
        key0 = number[0]
        target.append((key0, figures_info[key0][0], True))
        # Append matching figure
        key1 = figures_info[key0][2]
        if key1 == -1:
            print("WARNING: no match for number {}, key {}".format(number, key0))
        else:
            target.append((key1, figures_info[key1][0], False))
    target.append(("circle", circle_info[1], False))
    return target


if __name__ == "__main__":
    #args = parser()
    main()
