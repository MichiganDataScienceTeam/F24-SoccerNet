import math

def calculate_goalkeeper_influence(ball, toppost, bottompost, goal_center, goalkeeper):
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def angle_between(p1, p2, p3):
        # Compute angle at p2 between p1 and p3
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        return math.degrees(math.acos(dot / (mag_v1 * mag_v2)))
    
    def perpendicular_distance(p, line_start, line_end):
        # Distance from point p to line formed by line_start and line_end
        numerator = abs((line_end[1] - line_start[1]) * p[0] -
                        (line_end[0] - line_start[0]) * p[1] +
                        line_end[0] * line_start[1] - 
                        line_end[1] * line_start[0])
        denominator = math.sqrt((line_end[1] - line_start[1])**2 + 
                                (line_end[0] - line_start[0])**2)
        return numerator / denominator
    
    # Angular coverage by goalkeeper
    angle_coverage = (angle_between(toppost, goalkeeper, ball) +
                      angle_between(bottompost, goalkeeper, ball))
    
    # Goalkeeper alignment with goal center
    alignment_distance = perpendicular_distance(goalkeeper, ball, goal_center)
    
    return angle_coverage, alignment_distance

def defender_blocking_effect(ball, defenders, goalposts):
    effect = 0
    for defender in defenders:
        # Calculate the distance from defender to the ball
        distance_to_ball = math.sqrt((ball[0] - defender[0])**2 + (ball[1] - defender[1])**2)
        
        # Check if the defender is in line with the ball and the goalposts (based on angle)
        angle_to_goal = math.degrees(math.atan2(goalposts[0][1] - ball[1], goalposts[0][0] - ball[0]))
        defender_angle = math.degrees(math.atan2(defender[1] - ball[1], defender[0] - ball[0]))
        
        # Calculate how much the defender blocks the shot by considering the angle difference
        angle_difference = abs(angle_to_goal - defender_angle)
        
        # If the defender is close to the ball and aligned with the goal, they will block the shot
        if distance_to_ball < 15 and angle_difference < 20:  # Close and well-aligned defenders
            effect += 0.05  # Increase impact for close and aligned defenders
    return effect

def calculate_xg(ball, goal_center, goal_left, goal_right, goalkeeper, defenders, angle, weights):
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # Compute smarter goalkeeper influence
    angle_coverage, alignment_distance = calculate_goalkeeper_influence(
        ball, goal_left, goal_right, goal_center, goalkeeper
    )
    
    # Calculate defender blocking effect
    defender_penalty = defender_blocking_effect(ball, defenders, (goal_left, goal_right))
    
    # Extract weights
    w1, w2, w3, w4, w5 = weights
    
    # Distance to goal center
    d = distance(ball, goal_center)
    
    # Logistic regression combining all factors
    z = (w1 * d +
         w2 * angle +
         w3 * angle_coverage +
         w4 * alignment_distance +
         w5 * defender_penalty)  # Use defender penalty based on blocking effect
    
    # Apply sigmoid function to get xG
    xg = 1 / (1 + math.exp(-z))
    return xg

# Example usage
ball = (18, 12)  # Position of the ball
goal_left = (0, 8)  # Top-left corner of the goal
goal_right = (0, 0)  # Bottom-left corner of the goal
goal_center = (0, 4)  # Center of the goal
goalkeeper = (2, 7)  # Position of the goalkeeper
defenders = []  # Example defender positions
angle = 25.9  # Already computed angle between ball and goalposts

# New weights with a higher influence for defender effect
weights = [-0.1, 0.099, -0.01, -0.015, -6]  # Increase the weight of defender effect

# Calculate xG
xg = calculate_xg(ball, goal_center, goal_left, goal_right, goalkeeper, defenders, angle, weights)
print(f"Expected Goals (xG): {xg:.2f}")
ball = (1, 4)  # Position of the ball
goal_left = (0, 8)  # Top-left corner of the goal
goal_right = (0, 0)  # Bottom-left corner of the goal
goal_center = (0, 4)  # Center of the goal
goalkeeper = (18, 4)  # Position of the goalkeeper
defenders = []  # Example defender positions
angle = 25.9  # Already computed angle between ball and goalposts
xg = calculate_xg(ball, goal_center, goal_left, goal_right, goalkeeper, defenders, angle, weights)
print(f"Expected Goals (xG): {xg:.2f}")
ball = (10, 4)  # Position of the ball
goal_left = (0, 8)  # Top-left corner of the goal
goal_right = (0, 0)  # Bottom-left corner of the goal
goal_center = (0, 4)  # Center of the goal
goalkeeper = (18, 4)  # Position of the goalkeeper
defenders = []  # Example defender positions
angle = 25.9  # Already computed angle between ball and goalposts
xg = calculate_xg(ball, goal_center, goal_left, goal_right, goalkeeper, defenders, angle, weights)
print(f"Expected Goals (xG): {xg:.2f}")
ball = (9, -5)  # Position of the ball
goal_left = (0, 8)  # Top-left corner of the goal
goal_right = (0, 0)  # Bottom-left corner of the goal
goal_center = (0, 4)  # Center of the goal
goalkeeper = (0.5, 0.5)  # Position of the goalkeeper
defenders = []  # Example defender positions
angle = 25.9  # Already computed angle between ball and goalposts
xg = calculate_xg(ball, goal_center, goal_left, goal_right, goalkeeper, defenders, angle, weights)
print(f"Expected Goals (xG): {xg:.2f}")
