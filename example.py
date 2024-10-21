# test_script.py

import SoccerAnalysis #eplace with your actual package name

# Call a function from your package
frames = SoccerAnalysis.process_video(
    source_video_path="20SecGoodVid.mov",
    target_video_path="interpolation_test4.mp4",
    player_api_key="",
    player_stub_path="stub_tracks_mid.pkl",
    player_project_name="football-players-detection-3zvbc",
    version_number=1,
    boxes_api_key="",
    boxes_project_name="football0detections",
    stub_frames_path="frames_stub.pkl",
    field_length=105,
    field_width=68
)

print("Process complete! Frames returned:", len(frames))
