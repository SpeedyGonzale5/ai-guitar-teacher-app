import cv2
import mediapipe as mp
import numpy as np
import json
from datetime import datetime
import textwrap
import time

# Configuration for different video/JSON combinations
CONFIGS = {
    1: {
        'json_file': 'pickleball.json',
        'video_file': 'pickleball_demo.mov',
        'output_file': 'pickleball_final.mp4'
    },
    2: {
        'json_file': 'pickleball2.json', 
        'video_file': 'ai_pickleball_2v2.mp4',
        'output_file': 'pickleball2_final.mp4'
    },
    3: {
        'json_file': 'pickleball3.json',
        'video_file': 'ai_pickleball_1v1.mp4', 
        'output_file': 'pickleball3_final.mp4'
    }
}

# Choose which configuration to use (change this number: 1, 2, or 3)
CURRENT_CONFIG = 3
config = CONFIGS[CURRENT_CONFIG]

# Load pickleball data from JSON
try:
    with open(config['json_file'], 'r') as f:
        pickleball_data = json.load(f)
    print(f"Successfully loaded {config['json_file']}")
except Exception as e:
    print(f"Error loading JSON file {config['json_file']}: {e}")
    exit(1)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the video files
process_video_path = config['video_file']
display_video_path = config['video_file']

# Open processing video (lower res)
process_cap = cv2.VideoCapture(process_video_path)
if not process_cap.isOpened():
    print(f"Error: Could not open video file {process_video_path}")
    exit(1)

process_fps = int(process_cap.get(cv2.CAP_PROP_FPS))
process_width = int(process_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
process_height = int(process_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video loaded: {process_video_path}")
print(f"FPS: {process_fps}, Resolution: {process_width}x{process_height}")

# Open display video (higher res)
display_cap = cv2.VideoCapture(display_video_path)
if not display_cap.isOpened():
    print(f"Error: Could not open display video file {display_video_path}")
    exit(1)

display_fps = int(display_cap.get(cv2.CAP_PROP_FPS))
display_width = int(display_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
display_height = int(display_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# List to store all processed frames
processed_frames = []

# Animation variables
last_point_time = None
animation_duration = 1.5  # seconds
current_color = (255, 255, 255)  # Start with white

def parse_timestamp(timestamp):
    # Convert timestamp (e.g., "0:05") to seconds
    minutes, seconds = timestamp.split(':')
    return float(minutes) * 60 + float(seconds)

def timestamp_to_frame(timestamp, fps):
    # Convert timestamp to frame number
    seconds = parse_timestamp(timestamp)
    return int(seconds * fps)

def wrap_text(text, font, scale, thickness, max_width):
    # Calculate the maximum number of characters that can fit in max_width
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        text_size = cv2.getTextSize(test_line, font, scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def get_animation_color(elapsed_time, is_winner):
    if elapsed_time >= animation_duration:
        return (255, 255, 255)  # Return to white
    
    # Calculate progress through animation (0 to 1)
    progress = elapsed_time / animation_duration
    
    if progress < 0.5:
        # Fade to color
        if is_winner:
            # Fade to green (BGR format)
            return (
                int(255 * (1 - progress * 2)),  # B
                255,                            # G
                int(255 * (1 - progress * 2))   # R
            )
        else:
            # Fade to red (BGR format)
            return (
                int(255 * (1 - progress * 2)),  # B
                int(255 * (1 - progress * 2)),  # G
                255                             # R
            )
    else:
        # Fade back to white
        if is_winner:
            # Fade from green to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                255,                                # G
                int(255 * ((progress - 0.5) * 2))   # R
            )
        else:
            # Fade from red to white
            return (
                int(255 * ((progress - 0.5) * 2)),  # B
                int(255 * ((progress - 0.5) * 2)),  # G
                255                                 # R
            )

# Convert timestamps to frame numbers and add feedback display duration
shots_key = 'shots' if 'shots' in pickleball_data else 'points'
for shot in pickleball_data[shots_key]:
    shot['frame_number'] = timestamp_to_frame(shot['timestamp_of_outcome'], process_fps)
    shot['feedback_end_frame'] = shot['frame_number'] + (5 * process_fps)  # Show feedback for 5 seconds

last_head = None
frame_count = 0
process_every_n_frames = int(process_fps / 20)  # Process at 3 fps
last_point_result = None

# Initialize scores
team1_score = 0
team2_score = 0
serving_team = 1  # 1 or 2

print(f"Processing pickleball video: {config['video_file']} with data from {config['json_file']}...")

while process_cap.isOpened() and display_cap.isOpened():
    # Read frames from both videos
    process_ret, process_frame = process_cap.read()
    display_ret, display_frame = display_cap.read()
    
    if not process_ret or not display_ret:
        break

    frame_count += 1
    
    # Only process every nth frame
    if frame_count % process_every_n_frames == 0:
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Get the head landmark (landmark 0 is the top of the head)
            head = results.pose_landmarks.landmark[0]
            # Scale coordinates to display resolution
            head_x = int(head.x * display_width)
            head_y = int(head.y * display_height)
            last_head = (head_x, head_y)

    # Draw the arrow and name if we have a head position
    if last_head is not None:
        head_x, head_y = last_head
        arrow_height = 30
        arrow_width = 45
        arrow_tip_y = max(0, head_y - 110)
        # Triangle points for the arrow
        pt1 = (head_x, arrow_tip_y + arrow_height)  # tip
        pt2 = (head_x - arrow_width // 2, arrow_tip_y)  # left
        pt3 = (head_x + arrow_width // 2, arrow_tip_y)  # right
        pts = np.array([pt1, pt2, pt3], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(display_frame, [pts], (0, 0, 255))  # Red arrow
        # Draw the name above the arrow with black border
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "Pedro"
        text_size = cv2.getTextSize(text, font, 2.5, 6)[0]
        text_x = head_x - text_size[0] // 2
        text_y = arrow_tip_y - 10
        # Black border
        cv2.putText(display_frame, text, (text_x, text_y), font, 2.5, (0, 0, 0), 15, cv2.LINE_AA)
        # White fill
        cv2.putText(display_frame, text, (text_x, text_y), font, 2.5, (255, 255, 255), 6, cv2.LINE_AA)

    # Calculate current shot statistics
    current_feedback = None
    current_shot_info = None
    
    for shot in pickleball_data[shots_key]:
        if shot['frame_number'] <= frame_count:
            if shot['frame_number'] == frame_count:
                # New shot detected
                last_point_time = time.time()
                
                # Handle different data structures
                if 'concluding_shot_player' in shot:
                    # Old structure
                    last_point_result = shot['point_winner']
                    current_shot_info = f"{shot['concluding_shot_player']} - {shot['concluding_shot_type']}"
                    
                    # Update scores based on the point result
                    if "Side Out" in shot['score_after_point']:
                        serving_team = 3 - serving_team
                    else:
                        if shot['point_winner'] == "Far Side Team":
                            team2_score += 1
                        else:
                            team1_score += 1
                else:
                    # New structure
                    last_point_result = shot['point_winner']
                    current_shot_info = f"{shot['shot_by_player']} - {shot['shot_type']}"
                    
                    # Update scores based on current_score
                    if "Side Out" in shot['current_score']:
                        serving_team = 3 - serving_team
                    else:
                        # Parse score from current_score (e.g., "3-2-1")
                        score_parts = shot['current_score'].split('-')
                        if len(score_parts) >= 2:
                            team1_score = int(score_parts[0])
                            team2_score = int(score_parts[1])
            
            # Check if we should show feedback for this shot
            if shot['frame_number'] <= frame_count <= shot['feedback_end_frame']:
                current_feedback = shot['feedback']

    # Display pickleball statistics in top left
    stats_font = cv2.FONT_HERSHEY_SIMPLEX
    stats_border = (0, 0, 0)  # Black border
    stats_scale = 2.1
    stats_thickness = 6
    stats_border_thickness = 12
    stats_spacing = 90
    white_color = (255, 255, 255)

    # Position for stats (top left with padding)
    stats_x = 30
    stats_y = 150

    # Calculate animation color if needed
    if last_point_time is not None:
        elapsed_time = time.time() - last_point_time
        if elapsed_time < animation_duration:
            current_color = get_animation_color(elapsed_time, last_point_result == "Foreground Team")
        else:
            current_color = white_color
            last_point_time = None

    # Draw team scores
    team1_text = f"Team 1: {team1_score}"
    team2_text = f"Team 2: {team2_score}"
    serving_text = f"Serving: Team {serving_team}"
    
    # Draw team 1 score with border
    cv2.putText(display_frame, team1_text, 
                (stats_x, stats_y), stats_font, stats_scale, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    # Draw team 1 score with color
    cv2.putText(display_frame, team1_text, 
                (stats_x, stats_y), stats_font, stats_scale, 
                current_color if last_point_result == "Foreground Team" else white_color, stats_thickness, cv2.LINE_AA)
    
    # Draw team 2 score
    cv2.putText(display_frame, team2_text, 
                (stats_x, stats_y + stats_spacing), stats_font, stats_scale, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    cv2.putText(display_frame, team2_text, 
                (stats_x, stats_y + stats_spacing), stats_font, stats_scale, 
                current_color if last_point_result == "Far Side Team" else white_color, stats_thickness, cv2.LINE_AA)
    
    # Draw serving team
    cv2.putText(display_frame, serving_text, 
                (stats_x, stats_y + stats_spacing * 2), stats_font, stats_scale * 0.8, 
                stats_border, stats_border_thickness, cv2.LINE_AA)
    cv2.putText(display_frame, serving_text, 
                (stats_x, stats_y + stats_spacing * 2), stats_font, stats_scale * 0.8, 
                (255, 255, 0), stats_thickness, cv2.LINE_AA)  # Yellow for serving

    # Display shot information if available
    if current_shot_info:
        shot_font = cv2.FONT_HERSHEY_SIMPLEX
        shot_scale = 1.5
        shot_color = (255, 255, 255)
        shot_border = (0, 0, 0)
        shot_thickness = 4
        shot_border_thickness = 8
        
        # Position shot info in top right
        shot_x = display_width - 600
        shot_y = 150
        
        # Draw shot info with border
        cv2.putText(display_frame, current_shot_info, 
                    (shot_x, shot_y), shot_font, shot_scale, 
                    shot_border, shot_border_thickness, cv2.LINE_AA)
        cv2.putText(display_frame, current_shot_info, 
                    (shot_x, shot_y), shot_font, shot_scale, 
                    shot_color, shot_thickness, cv2.LINE_AA)

    # Display feedback if available
    if current_feedback:
        feedback_font = cv2.FONT_HERSHEY_SIMPLEX
        feedback_scale = 1.2  # Larger text for better visibility
        feedback_color = (255, 255, 255)  # White text
        feedback_shadow = (0, 0, 0)  # Black shadow
        feedback_thickness = 3
        feedback_spacing = 45  # More spacing for larger text

        # Wrap text to fit within 50% of screen width (narrower for modern captions)
        max_width = int(display_width * 0.5)
        wrapped_lines = wrap_text(current_feedback, feedback_font, feedback_scale, feedback_thickness, max_width)

        # Calculate total height of wrapped text
        total_height = len(wrapped_lines) * feedback_spacing
        start_y = display_height - 120 - total_height  # Move feedback higher up from bottom

        # Draw each line centered with shadow effect
        for i, line in enumerate(wrapped_lines):
            text_size = cv2.getTextSize(line, feedback_font, feedback_scale, feedback_thickness)[0]
            feedback_x = (display_width - text_size[0]) // 2
            feedback_y = start_y + (i * feedback_spacing)

            # Draw shadow (offset by 2 pixels)
            cv2.putText(display_frame, line, 
                        (feedback_x + 2, feedback_y + 2), feedback_font, feedback_scale, 
                        feedback_shadow, feedback_thickness, cv2.LINE_AA)
            # Draw main text
            cv2.putText(display_frame, line, 
                        (feedback_x, feedback_y), feedback_font, feedback_scale, 
                        feedback_color, feedback_thickness, cv2.LINE_AA)

    # Store the processed frame
    processed_frames.append(display_frame.copy())

    # Display the frame (optional)
    cv2.imshow('Pickleball Analysis', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
process_cap.release()
display_cap.release()
cv2.destroyAllWindows()

print("Creating final pickleball video...")

# Create the final video at normal speed
final_output_path = config['output_file']
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
final_out = cv2.VideoWriter(final_output_path, fourcc, display_fps, (display_width, display_height))

# Write all frames to the final video
for frame in processed_frames:
    final_out.write(frame)

final_out.release()

print(f"Processing complete. Final pickleball video saved to {final_output_path}")
print(f"To switch videos, change CURRENT_CONFIG to 1, 2, or 3 in the script") 