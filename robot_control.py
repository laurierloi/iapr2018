######## Import
import math
import numpy as np
from iapr.robot import Robot

######## Class definition
class RobotController :

    def __init__(self, name):
        self.name = name
        self.__a = []
        self.__b = []
        self.__a_init = 0
        self.__b_init = 0
        self.robot = Robot(hostname='ev3dev.local')
        #print('a = {} AND b = {} \n'.format(self.__a, self.__b))

    def convAngle(self, theta, x_sign) :
        if x_sign >= 0 :
            convert_angle = theta
        else :
            if theta >= 0 :
                convert_angle = math.radians(180) - theta
            else :
                convert_angle = -math.radians(180) - theta
        return convert_angle


    def calibration(self) :
        dist = 20
        angle = 45

        # Rotate robot of 45 degrees to calibrate the future movements
        self.robot.steer_in_place(angle=math.degrees(angle))

        # Move robot of 10cm to calibrate the future movements
        self.robot.move_forward(distance=dist)

    def calcFactorCalibr(self, x_i, y_i, theta_i, x_f, y_f, theta_f) :
        # Calculate the corrections factors based on the calibration
        angle = 45
        corr_a = math.radians(angle)/(theta_f-theta_i)

        dist_t = 10
        dist_p = np.linalg.norm([x_f - x_i, y_f - y_i])
        corr_b = dist_t/dist_p

        self.__a_init = np.abs(corr_a)
        self.__b_init = np.abs(corr_b)

    def calcDisplacement(self, x_i, y_i, theta_i, x_f, y_f) :
        dir_vector = [x_f-x_i, y_f-y_i]
        dist = np.linalg.norm(dir_vector)

        # Calculate angle of this vector
        if dir_vector[0] >= 0 :
            if dir_vector[1] >= 0 :
                angle = - math.acos(dir_vector[0]/dist)
            else :
                angle = np.pi - math.acos(dir_vector[0]/dist)
        else :
            if dir_vector[1] >= 0 :
                angle = math.acos(dir_vector[0]/dist) - np.pi
            else :
                angle = math.acos(dir_vector[0]/dist)
        angle = -(theta_i + angle)

        return angle, dist

    def corrFactors(self, x_p, y_p, theta_p) :
        x_i = self.x_prev
        y_i = self.y_prev
        x_t = self.x_t
        y_t = self.y_t
        theta_t = self.theta_t

        # Calculate the correction factors after each displacement
        corr_a = theta_t/theta_p

        #dist_t = np.linalg.norm([x_t - x_i, y_t - y_i])
        dist_p = np.linalg.norm([x_p - x_i, y_p - y_i])
        corr_b = dist_t/dist_p

        self.__a.append(np.abs(corr_a))
        self.__b.append(np.abs(corr_b))

    def checkOnTheShape(self, x_r, y_r, is_number, finish = False,dist_min=20) :
        x_shape = self.x_t
        y_shape = self.y_t
        dist = np.linalg.norm([x_r - x_shape, y_r - y_shape])

        if dist <= dist_min :
            #Beep the robot
            if finish == True :
                self.robot.beep(count=3)
            elif is_number == True :
                self.robot.beep(count=1)
            else :
                self.robot.beep(count=2)
            return True
        else :
            return False

    def GoTo(self, x_i, y_i, theta_i, x_f, y_f) :
        corr_fact_angle = 0
        corr_fact_dist = 0

        # Calculate theoritical movements of the robot
        angle, dist = self.calcDisplacement(x_i, y_i, theta_i, x_f, y_f)

        self.x_prev = x_i
        self.y_prev = y_i
        self.x_t = x_f
        self.y_t = y_f
        self.theta_t = angle + theta_i
        self.dist_t = dist

        # Calculate average of correction factors
        if len(self.__a) > 0 :
            corr_fact_angle = self.__a_init + np.mean(self.__a)
        else :
            corr_fact_angle = self.__a_init

        if len(self.__b) > 0 :
            corr_fact_dist = self.__b_init + np.mean(self.__b)
        else :
            corr_fact_dist = self.__b_init

        print('a = {} AND b = {}'.format(corr_fact_angle, corr_fact_dist))

        # Apply the movements using the correction factors
        # Rotate robot
        self.robot.steer_in_place(angle=math.degrees(corr_fact_angle*angle))

        # Move robot
        self.robot.move_forward(distance=corr_fact_dist*dist)

######### Code
# Create Robot instance
#robot = Robot(hostname='ev3dev.local')

def main():
    # Create robotControl instance
    robotControl = RobotController('Wall-E')
    x_i = 100
    y_i = 100
    theta_i = math.radians(-45)

    x_f = 120
    y_f = 100
    theta_f = 0

    robotControl.calibration()
    robotControl.calcFactorCalibr(x_i, y_i, theta_i, x_f, y_f, theta_f)

    robotControl.GoTo(x_f, y_f, theta_f, 150, 150)

    x_f = 150
    y_f = 150

    #robotControl.convAngle(-np.pi/4, 1)

    #robotControl.GoTo(robot, x_i,y_i, theta_i, x_f, y_f)
    #robotControl.GoTo(x_i,y_i, theta_i, x_f, y_f)

    robotControl.checkOnTheShape(100, 100, 100, 110, True)
