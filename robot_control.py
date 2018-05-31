######## Import
import math
import numpy as np
from iapr.robot import Robot

######## Class definition
class RobotController :

    def __init__(self):
        self.__a = []
        self.__b = []
        self.__a_init = 1
        self.__b_init = 1
        self.cal_dist = 20
        self.cal_angle = 45
        self.robot = Robot(hostname='ev3dev.local')
        #print('a = {} AND b = {} \n'.format(self.__a, self.__b))

    def convAngle(self, theta, x_sign) :
        if x_sign >= 0 :
            convert_angle = theta
        else :
            if theta >= 0 :
                convert_angle = math.radians(180) - theta
            else :
                convert_angle = -theta - math.radians(180)
        return convert_angle


    def calibration(self) :

        # Rotate robot of 45 degrees to calibrate the future movements
        self.robot.steer_in_place(angle=self.cal_angle)

        # Move robot of 20cm to calibrate the future movements
        self.robot.move_forward(distance=self.cal_dist)

    def calcFactorCalibr(self, x_i, y_i, theta_i, x_f, y_f, theta_f) :
        # Calculate the corrections factors based on the calibration
        angle = self.cal_angle
        corr_a = math.radians(angle)/(theta_f-theta_i)

        dist_t = self.cal_angle
        dist_p = np.linalg.norm([x_f - x_i, y_f - y_i])
        corr_b = dist_p/dist_t

        self.__a_init = np.abs(corr_a)
        self.__b_init = np.abs(corr_b)
        print("Calibration corrections: a:{}, b:{}".format(self.__a_init, self.__b_init))

    def calcDisplacement(self, x_i, y_i, theta_i, x_f, y_f) :
        dir_vector = [x_f-x_i, np.abs(y_f-y_i)]
        dist = np.linalg.norm(dir_vector)

        # Calculate angle of this vector
        if dir_vector[0] >= 0 :
            if dir_vector[1] >= 0 :
                angle = - math.asin(dir_vector[1]/dist)
            else :
                angle = np.pi - math.asin(dir_vector[1]/dist)
        else :
            if dir_vector[1] >= 0 :
                angle = math.asin(dir_vector[1]/dist) - np.pi
            else :
                angle = math.asin(dir_vector[1]/dist)
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

    def GoTo(self, x_i, y_i, theta_i, x_f, y_f, epsilon = 4) :
        print("X_i:{}, Y_i:{}, Theta_i:{}, x_f:{}, y_f:{}".format(x_i, y_i, theta_i, x_f, y_f))
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
            corr_fact_dist = (self.__b_init + np.mean(self.__b))/self.cal_dist
        else :
            corr_fact_dist = (self.__b_init)/self.cal_dist
        #corr_fact_angle = 1
        #corr_fact_dist = 2

        print('Corr factors: a = {} AND b = {}'.format(corr_fact_angle, corr_fact_dist))

        # Calculate the movements using the correction factors
        angle_to_move = math.degrees(corr_fact_angle*angle)
        distance_to_move = corr_fact_dist*dist
        print("Movements: angle: {}, distance: {}".format(angle_to_move, distance_to_move))

        if np.abs(angle_to_move) > epsilon:
            # Rotate robot
            self.robot.steer_in_place(angle=angle_to_move)
        else:
            # Move robot
            self.robot.move_forward(distance=distance_to_move)

    def check_angle(self, theta_i, epsilon=0.1):
        if  (self.theta_t - epsilon) < theta_i and (self.theta_t + epsilon) > theta_i:
            return True
        else:
            return false


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
