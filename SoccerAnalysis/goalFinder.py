import bboxUtils
import pandas as pd
import numpy as np

# goal_detector.py

class Goal:
    def __init__(self, field_width, field_height, tracks):
        """
        Initialize the Goal object with field dimensions and goal areas.
        
        :param field_width: Width of the field in pixels
        :param field_height: Height of the field in pixels
        :param left_goal_coords: Tuple ((x1, y1), (x2, y2)) for the left goal area
        :param right_goal_coords: Tuple ((x1, y1), (x2, y2)) for the right goal area
        """
        self.field_width = field_width
        self.field_height = field_height
        self.goal_detected = False
        self.goal_time = None
        self.goalMid = (self.field_width)/2
        self.bottomGoal = self.goalMid - 4 #convert to meters
        self.topGoal = self.goalMid + 4 #convert to meters
        self.tracks = tracks
        tracks["goals"] = 0
    def is_ball_in_goal_zone(self, ball_position):
        """
        Check if the ball is within either goal zone.
        
        :param ball_position: Tuple (x, y) of the ball's coordinates
        :return: 'left', 'right', or None depending on the goal area
        """
        x, y = ball_position
        if (x == 0 and ((y <= self.topGoal) or (y>=self.bottomGoal))):
            return 'left'
        elif (x == self.field_width and ((y <= self.topGoal) or (y>=self.bottomGoal))):
            return 'right'
        return None

    def detect_goal(self, frame_index, ball_position, frame_rate):
        """
        Detect if a goal is scored by checking ball position.
        
        :param frame_index: The current frame number
        :param ball_position: Tuple (x, y) of the ball's coordinates
        :param frame_rate: The frame rate of the video
        :return: None
        """
        goal_side = self.is_ball_in_goal_zone(ball_position)
        
        if goal_side:
            self.goal_detected = True
            self.goal_time = frame_index / frame_rate
            print(f"Goal detected in {goal_side} goal at {self.goal_time:.2f} seconds")
            self.tracks["goals"] += 1 
            
    def reset(self):
        """
        Reset the goal detection status for a new video sequence or frame set.
        """
        self.goal_detected = False
        self.goal_time = None

    def find_goals_in_tracks(self, tracks, frame_rate):
        """
        Loops through frames in tracks to detect goals.
        
        :param tracks: List of tracking data, each entry containing frame data
        :param frame_rate: The frame rate of the video
        """
        for frame_index, frame_data in enumerate(tracks):
            ball_position = frame_data.get('ball')
            
            # Check if ball position is detected
            if ball_position:
                # Detect goal
                self.detect_goal(frame_index, ball_position, frame_rate)
                
                # Stop searching after the first detected goal if needed
                if self.goal_detected:
                    break
