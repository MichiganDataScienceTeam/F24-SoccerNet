# SoccerNet
<table>
  <tr>
    <td>
      <img src="assets/icon.jpeg" width="1500">
    </td>
    <td style="text-align: left; padding-left: 20px;">
By harnessing the power of computer vision, we can transform the way soccer can be understood, watched, and even played. This project uses pre-trained Computer Vision models to analyze every moment of a match—tracking players, speeds, ball control, and more—to provide deep, actionable stats. Although no past ML experience is required, members should be very comfortable with Python as the overall goal of the project will be to build a Python package for soccer match analysis.
    </td>
  </tr>
</table>

## Key Features

### 1. Camera Calibration, Pitch Localization, and Homography
We use a combination of camera calibration, pitch localization, and homography to ensure accurate alignment of the video frames with real-world coordinates. By detecting key stationary points on the soccer pitch, we calibrate the camera and localize the pitch within the video. This process allows us to ground all measurements, such as player positions and movements, in real-world field locations. Additionally, by computing a homography from multiple key points on the pitch, we map 2D video data to 3D space, enabling highly precise spatial analysis and ensuring that our tracking and analysis are accurate throughout the game.

### 2. Team Assignments
Players are automatically assigned to teams based on their position and movement on the field. This allows us to derive team-based statistics, formations, and strategies.

### 3. Player Speed and Distance Traveled
By tracking players over time, we can calculate their speed and total distance traveled during a match. This data provides insights into player stamina, effort, and performance.

## Example

<p align="center">
  <img src="/assets/example.gif" alt="Example" />
</p>


## How It Works
The system works by combining computer vision models with traditional field recognition techniques. The workflow includes:
- **Player Detection**: Using pre-trained Roboflow models to detect players, referees, and the ball in video frames.
- **Field Detection**: Using pre-trained Roboflow models to detect relevant field areas like half field and center circle.
- **Object Tracking**: Tracking detected objects across frames to generate consistent data for each player.
- **Homography Computation**: Mapping the video frames onto a real-world soccer pitch using detected key points for accurate spatial measurements.

## Future Enhancements
We are continually working to improve and expand the capabilities of this project, with the following planned features:
- Enhanced player detection and tracking accuracy.
- Additional metrics, including pass success rates and shot accuracy.
- Real-time analysis for live matches.

---

### Getting Started (In Developement)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MichiganDataScienceTeam/F24-SoccerNet.git

2. **Install dependencies: Make sure you have Python 3.11 installed before proceeding**:
   ```bash
   pip install -r requirements.txt

3. **install the package in editable mode**:
   ```bash
   pip install -e .

4. Run the analysis in example.py:
   

View results: Results will be saved as annotations and can be visualized in an output video through the provided tools.
