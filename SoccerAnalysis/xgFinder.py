import math
import calcxg
class xgFinder():
    def __init__(self, length, width):
        self.frame_window=5
        self.frame_rate=30
        self.field_length = length
        self.field_width = width

    def findGoalkeeper(self, tracks, framenum, right):
        # We are finding the most extreme player (goalkeeper)
        # if right is true that means we are attacking the right goal
        players = tracks["players"][framenum]
        # Problem we have - how do coordinates even work is the center 0,0 is the bottom left 0,0 -> regardless the smallest x value will be leftmost
        # And the largest x value will be right most
        mostextreme = players[0]["Position Transformed"]
        mostextremeindex = 0
        for i in range(len(players)):
            # Here we intend to iterate through all of the class_ids in tracks["players"][framenum], or in other words all of the players
            playerpos = players[i]["Position Transformed"]
            if(right):
                # If they are attacking the right goal, we check to see whether their x coordinate is greater to the right than our current max
                if playerpos[0] > mostextreme:
                    mostextreme = playerpos[0]
                    mostextremeindex = i
            else:
                # If they are attacking the left goal, we check to see whether their x coordinate is greater to the right than our current max
                if playerpos[0] < mostextreme:
                    mostextreme = playerpos[0]
                    mostextremeindex = i
        #mostextremeindex contains either the right or leftmost player. We return their indices with this statement
        return tracks["players"][framenum][mostextremeindex]["Position Transformed"]
    def inTriangle(TopPost,BottomPost,Ball,Defender):
        def cross_product(v1, v2):
            # Cross product of 2D vectors
            return v1[0] * v2[1] - v1[1] * v2[0]
    
        def vector(p1, p2):
            # Create a vector from p1 to p2
            return (p2[0] - p1[0], p2[1] - p1[1])
        PA = vector(Defender, TopPost)
        PB = vector(Defender, BottomPost)
        PC = vector(Defender, Ball)
        
        # Calculate cross products
        cross1 = cross_product(PA, PB)
        cross2 = cross_product(PB, PC)
        cross3 = cross_product(PC, PA)
        
        # Check if all cross products have the same sign
        return (cross1 > 0 and cross2 > 0 and cross3 > 0) or (cross1 < 0 and cross2 < 0 and cross3 < 0)

    def findDefenders(self, tracks, framenum, right, ball,goalkeeper, goalposts):
        # if right is true that means we are attacking the right goal
        # All the players in between ball and goal 
        # We are going to treat attackers and defenders as the same - so this is really just finding all players between
        
        players = tracks["players"][framenum]
        indices = []

        for i in range(len(players)):
            # This is meant to iterate through all of the class_ids in tracks["players"][framenum]
            # As far as we understand this should contain all of the players that are currently in the framenum
            playerpos = players[i]["Position Transformed"]
            # Skip the goalkeeper
            if playerpos == goalkeeper:
                continue
            if self.inTriangle(goalposts[1],goalposts[0],ball, playerpos):
                indices.append(i)
        return indices

    def findxg(self,goalkeeper, posts, ball, defenders, goal, angle):
        # We still have to figure this out
        # None of this thinking though utilizes/depends on the code, so we decided to push it off
        print(f"Defenders length: {len(defenders)}")
        print(f"Ball: {ball}")
        print(f"Top Goalpost: {posts[0]}")
        print(f"Bottom Goalpost: {posts[1]}") 
        print(f"Goalkeeper: {goalkeeper}")
        print(f"Goal: {goal}")
        print(f"Angle: {goalkeeper}")
        weights = [-0.1, 0.099, -0.01, -0.015, -6] 
        
        return calcxg.calculate_xg(ball, goal, posts[0], posts[1], goalkeeper, defenders, angle, weights)   
        
    def angle_between_vectors(A, B, C):
        def vector(p1, p2):
            # Create a vector from p1 to p2
            return (p2[0] - p1[0], p2[1] - p1[1])
        
        def dot_product(v1, v2):
            # Dot product of two vectors
            return v1[0] * v2[0] + v1[1] * v2[1]
        
        def magnitude(v):
            # Magnitude of a vector
            return math.sqrt(v[0]**2 + v[1]**2)
        
        # Vectors
        AC = vector(A, C)
        BC = vector(B, C)
        
        # Calculate the dot product and magnitudes
        dot = dot_product(AC, BC)
        mag_AC = magnitude(AC)
        mag_BC = magnitude(BC)
        
        # Avoid division by zero
        if mag_AC == 0 or mag_BC == 0:
            raise ValueError("One of the vectors has zero length, cannot compute angle.")
        
        # Calculate cosine of the angle
        cos_theta = dot / (mag_AC * mag_BC)
        
        # Clamp cos_theta to the range [-1, 1] to avoid numerical errors
        cos_theta = max(-1, min(1, cos_theta))
        
        # Calculate the angle in radians and convert to degrees
        angle_radians = math.acos(cos_theta)
        angle_degrees = math.degrees(angle_radians)
        
        return angle_degrees

    def findXGPoints(self,tracks, right):
        numframes = len(tracks[])
        # When finding the ball use frames around whatever number we calculate
        # We have numframes frames in our clip
        # To calculate the frames that we want to use we can just do some basic math on this and find evenly spaced values
        # So one problem is that we don't know frames to seconds, but we have this in self.framerate'
        
        # We are assuming any clips given to this model begin when you want to start assessing, and end as soon as the ball is struck
        fps = self.framerate
        numseconds = numframes / fps
        # Our clip is this long
        clips = [0]
        prog = 0.25
        prognum = 0.25 * fps
        while prog < numseconds:
            clips.append(prognum)
            prog +=0.25 
            prognum += 0.25*fps
        # clips now contains the frame numbers for all of the clips we are interested in (0.25 second intervals)
        # Maybe we cut the clip when he shoots
        # xgLog = []
        # This will be the xg of all of our clips in chronological order
        for i in range(clips):
            framenum = clips[i]
            # (x,y) = tracks[Player/ref/ball/][frame number][classid]["position transformed"]
            # Before this we will need code to determine when to start looking for xg - or that's too complicated and we just always do it in the clip
            # We can shorten their clip also with no problems 
            goalkeeper = self.findGoalkeeper(framenum, right)
            posts = ((0, self.field_width / 2 + 3.66),(0, self.field_width / 2 - 3.66) )
            if right:
                posts = ((self.field_length, self.field_width / 2 + 3.66),(self.field_length, self.field_width / 2 - 3.66) )

            ball = tracks["ball"][framenum][0]["Position Transformed"]
            # We do not know what to put for classid of ball - is it just 0?

            defenders = self.findDefenders(framenum,right, ball, goalkeeper, posts)
            goal = (0, self.field_width / 2)
            if right:
                goal = (self.field_length, self.field_width / 2)
            self.findxg(goalkeeper,posts, ball, defenders,goal, self.angle_between_vectors(posts[1], posts[0], ball))
            
        return 