from assign_team import TeamAssigner
import bboxUtils

#TODO: differentiate teams
#TODO: find position of the ball and players
#TODO: based on the position of the ball, find the player closest to it
#TODO: based on this, use our determined threshold to increment a variable indicating which team the ball is with during this frame




team_dict = {}
# team1 : num frames of team 1
# team2 : num frames of team 2




def ball_control(self, tracks, frames, track_id):
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for track_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track['bbox'], track_id)

            if team not in team_dict:
                team_dict[team] = 0



    for object, object_tracks in tracks.items():
        for frame_num, track in enumerate(object_tracks):
            for track_id, track_info in track.items():
                bbox = track_info['bbox']
                if object == 'ball':
                    position= bboxUtils.get_center_of_bbox(bbox)
                elif object == 'player' or object == 'referee':
                    position = bboxUtils.get_foot_position(bbox)


        # for each frame in video grab the postion of ball and player


    