#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 00:01:19 2019

@author: christian
"""

##  @package track_generation
#  Script to generate tracks with straight lines and arc
#

import matplotlib.pyplot as plt
import numpy as np
#import oyaml as yaml
from mpc.track.src import track_functions

##Houses information about a particular arc on a track
class arc:
    def __init__(self,x0,y0,radius,theta_s,theta_f,pointsPerArcLength):
        self.x0 = x0;
        self.y0 = y0;
        self.radius = radius;
        self.curvature = 1/radius
        self.theta_s = theta_s;
        self.theta_f = theta_f;
        self.pointsPerArcLength = pointsPerArcLength;
        self.length = abs(theta_f-theta_s)*radius;
        
##Houses information about a particular straight line on a track
class line:
    def __init__(self,x0,y0,x1,y1,pointsPerArcLength):
        self.x0 = x0;
        self.y0 = y0;
        self.x1 = x1;
        self.y1 = y1;
        self.pointsPerArcLength = pointsPerArcLength;
        self.length = np.sqrt((y1-y0)**2 + (x1-x0)**2);
    
##Main class which can be used to generate the track    
class trackGenerator:
    ##The constructor
    def __init__(self, density, track_width):
        self.lastSegmentType = "line";
        self.chainOfSegments  = np.array([]);
        self.xCoords = []
        self.yCoords = []
        self.xRate = []
        self.yRate = []
        self.tangentAngle = []
        self.arcLength = []
        self.curvature = []
        self.track_density = density
        self.track_width = track_width
    
    ##Chains a line to the list of segments    
    def addLine(self,x0,y0,x1,y1,numOfPoints):
        self.chainOfSegments = np.append(self.chainOfSegments,line(x0,y0,x1,y1,numOfPoints));
        
    ##Chains an arc to the list of segments
    def addArc(self,x0,y0,radius,theta_s,theta_f,numOfPoints):
        self.chainOfSegments = np.append(self.chainOfSegments,arc(x0,y0,radius,theta_s,theta_f,numOfPoints));
        
    ##After the track has been constructed points on the track can be generated using this method
    def populatePointsAndArcLength(self):
        for segment in self.chainOfSegments:
            numOfPoints = int(segment.pointsPerArcLength*segment.length);
            if type(segment) is line:
                for i in range(0,numOfPoints):
                    self.xCoords.append(segment.x0+(segment.x1-segment.x0)/numOfPoints*i)
                    self.yCoords.append(segment.y0+(segment.y1-segment.y0)/numOfPoints*i)
                    self.xRate.append((segment.x1-segment.x0)/numOfPoints)
                    self.yRate.append((segment.y1-segment.y0)/numOfPoints)
                    self.curvature.append(0.0)
                    if self.arcLength == []:
                        self.arcLength.append(0)
                    else:
                        self.arcLength.append(segment.length/numOfPoints+self.arcLength[-1])
            if type(segment) is arc:
                for i in range(0,numOfPoints):    
                    self.xCoords.append(segment.x0+segment.radius*np.cos((segment.theta_f - segment.theta_s)/numOfPoints*i+segment.theta_s))
                    self.yCoords.append(segment.y0+segment.radius*np.sin((segment.theta_f - segment.theta_s)/numOfPoints*i+segment.theta_s))
                    if (segment.theta_f - segment.theta_s) > 0:
                        self.curvature.append(1/segment.radius)
                    else:
                        self.curvature.append(-1/segment.radius)
                    if (segment.theta_f > segment.theta_s):
                        self.xRate.append(-np.sin((segment.theta_f - segment.theta_s)/numOfPoints*i+segment.theta_s))
                        self.yRate.append(np.cos((segment.theta_f - segment.theta_s)/numOfPoints*i+segment.theta_s))
                    else:
                        self.xRate.append(np.sin((segment.theta_f - segment.theta_s)/numOfPoints*i+segment.theta_s))
                        self.yRate.append(-np.cos((segment.theta_f - segment.theta_s)/numOfPoints*i+segment.theta_s))
                        
                    if self.arcLength == []:
                        self.arcLength.append(0)
                    else:
                        self.arcLength.append(segment.length/numOfPoints+self.arcLength[-1])
                
        self.xCoords = np.array(self.xCoords)
        self.xRate = np.array(self.xRate)
        self.yCoords = np.array(self.yCoords)
        self.yRate = np.array(self.yRate)
        self.arcLength = np.array(self.arcLength)
        self.tangentAngle = np.arctan2(self.yRate,self.xRate)
        norm_rate = np.sqrt(self.xRate*self.xRate+self.yRate*self.yRate)
        self.xRate = self.xRate / norm_rate
        self.yRate = self.yRate / norm_rate
        self.curvature = np.array(self.curvature)

        self.xCoords = np.append(self.xCoords,self.xCoords)
        self.yCoords = np.append(self.yCoords,self.yCoords)
        self.xRate = np.append(self.xRate,self.xRate)
        self.yRate = np.append(self.yRate,self.yRate)
        self.arcLength = np.append(self.arcLength, self.arcLength + self.arcLength[-1] + self.arcLength[1])
        self.tangentAngle = np.append(self.tangentAngle,self.tangentAngle)
        self.curvature = np.append(self.curvature, self.curvature)
        
    ##Writes the track to a yaml file, specified by path
    #def writePointsToYaml(self,path,density):
    #    temp_dict = {
    #            "xCoords" : self.xCoords.tolist(),
    #            "yCoords" : self.yCoords.tolist(),
    #            "xRate" : self.xRate.tolist(),
    #            "yRate" : self.yRate.tolist(),
    #            "tangentAngle" : self.tangentAngle.tolist(),
    #            "arcLength" : self.arcLength.tolist(),
    #            "curvature" : self.curvature.tolist(),
    #            "trackLength": self.arcLength.tolist()[-1]/2,
    #            "trackWidth": self.track_width,
    #            "density" : density,
    #            "x_init" : self.xCoords.tolist()[0],
    #            "y_init" : self.yCoords.tolist()[0],
    #           "yaw_init" : self.tangentAngle.tolist()[0]
    #            }
    #    top_dict = {"track" : temp_dict}
    #    with open(path, 'w') as outfile:
    #        yaml.dump(top_dict, outfile, default_flow_style=False)
                        
    ##Centers the track using infinity norms
    def centerTrack(self):
        mean_x = (np.max(self.xCoords)+np.min(self.xCoords))/2.0
        mean_y = (np.max(self.yCoords)+np.min(self.yCoords))/2.0
        self.xCoords = self.xCoords - mean_x
        self.yCoords = self.yCoords - mean_y
                           
    ##Returns a point at a certain arc length
    def pointAtArcLength(self,arcLength):
        temp = self.arcLength - arcLength;
        idx = np.where(temp>0)
        plt.plot(self.xCoords[idx[0][0]],self.yCoords[idx[0][0]],'ro',label = 'Point at arc length ' + str(arcLength),markersize=1)
        plt.legend()
        
    ##Plots the track
    def plotPoints(self, ax):
        #ax.plot(self.xCoords,self.yCoords,'o', markersize=0.05)
        mean_x = (np.max(self.xCoords)+np.min(self.xCoords))/2.0
        mean_y = (np.max(self.yCoords)+np.min(self.yCoords))/2.0
        #ax.plot(mean_x,mean_y,'go',label='Center',markersize=1)
        ax.plot(self.xCoords + self.yRate*self.track_width/2,self.yCoords-self.xRate*self.track_width/2,'ko',markersize=0.3)
        ax.plot(self.xCoords - self.yRate*self.track_width/2,self.yCoords+self.xRate*self.track_width/2,'ko',markersize=0.3)
        #ax.legend()
        ax.set_xlabel("Position x [m]")
        ax.set_ylabel("Position y [m]")
        ax.set_aspect('equal', 'box')
        
    ##Plots the direction of rate of change of each point on the track
    def plotDir(self, ax):
        ax.plot(self.xCoords+self.xRate,self.yCoords + self.yRate,'o',label='Rate Direction', markersize=1)
        
    def straight(self, init_state, dist):
        # Function that can be used to generate a straight line
        # of length dist from init_state
        x0 = init_state[0]
        y0 = init_state[1]
        direction = init_state[2]
        x1 = x0 + np.cos(direction)*dist
        y1 = y0 + np.sin(direction)*dist
        self.addLine(x0, \
                y0, \
                x1, \
                y1, \
                self.track_density)
        new_state = [x1, y1, direction]
        return new_state
    
    def left_turn(self, init_state, radius, turn_angle):
        # Function that can be used to generate a left turn
        # with specified radius of curvature and turn_angle
        x0 = init_state[0]
        y0 = init_state[1]
        direction = init_state[2]
        center_direction = direction + np.pi/2
        
        xc = x0 + np.cos(center_direction)*radius
        yc = y0 + np.sin(center_direction)*radius
        
        opposite_center_direction = center_direction + np.pi
        self.addArc(xc, \
                    yc, \
                    radius, \
                    opposite_center_direction, \
                    opposite_center_direction + turn_angle, \
                    self.track_density)
        x1 = xc + np.cos(opposite_center_direction + turn_angle)*radius
        y1 = yc + np.sin(opposite_center_direction + turn_angle)*radius
        direction1 = direction + turn_angle
        new_state = [x1, y1, direction1]
        return new_state
    
    def right_turn(self, init_state, radius, turn_angle):
        # Function that can be used to generate a right turn
        # with specified radius of curvature and turn_angle
        x0 = init_state[0]
        y0 = init_state[1]
        direction = init_state[2]
        center_direction = direction - np.pi/2
        xc = x0 + np.cos(center_direction)*radius
        yc = y0 + np.sin(center_direction)*radius
        
        opposite_center_direction = center_direction + np.pi
        self.addArc(xc, \
                    yc, \
                    radius, \
                    opposite_center_direction, \
                    opposite_center_direction - turn_angle, \
                    self.track_density)
        x1 = xc + np.cos(opposite_center_direction - turn_angle)*radius
        y1 = yc + np.sin(opposite_center_direction - turn_angle)*radius
        direction1 = direction - turn_angle
        new_state = [x1, y1, direction1]
        return new_state
        
        
def main():    
    
    track_density = 300
    track_width = 0.5
    gen = trackGenerator(track_density,track_width)
    track_name = 'DEMO_TRACK'
    
    t = 0.3
    init = [0,0,0]

    track_function = {
        'DEMO_TRACK'    : track_functions.demo_track,
        'HARD_TRACK'    : track_functions.hard_track,
        'LONG_TRACK'    : track_functions.long_track,
        'LUCERNE_TRACK' : track_functions.lucerne_track,
        'BERN_TRACK'    : track_functions.bern_track,
        'INFINITY_TRACK': track_functions.infinity_track,
        'SNAIL_TRACK'   : track_functions.snail_track
    }.get(track_name, track_functions.demo_track)
    
    track_function(gen, t, init)
    
    gen.populatePointsAndArcLength()
    gen.centerTrack()

    _, ax = plt.subplots(1,1)
    gen.plotPoints(ax)
    gen.pointAtArcLength(0)
    #gen.writePointsToYaml('../tracks/' + track_name + '.yaml', track_density)

    print('x_init: ' + str(gen.xCoords[0]))
    print('y_init: ' + str(gen.yCoords[0]))
    print('yaw_init: ' + str(gen.tangentAngle[0]))
    print('Total Arc Length: ' + str(gen.arcLength[-1]/2))
    plt.show()

if __name__ == "__main__":
   main()
      

        
