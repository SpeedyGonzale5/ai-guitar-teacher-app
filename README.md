# AI Pickleball Analysis Tool 🏓

An AI-powered pickleball analysis tool that overlays scores, shot info, and coaching feedback on top of your pickleball videos using OpenCV + MediaPipe.

## Features

- **Real-time Player Tracking**: Uses MediaPipe pose detection to track player movements
- **Score Tracking**: Automatically updates and displays team scores
- **Shot Analysis**: Displays shot types and player actions with timing
- **Animated Feedback**: Color-coded visual feedback for point outcomes
- **Multiple Video Support**: Configure different video/analysis combinations
- **Professional Overlay**: Clean, broadcast-style graphics overlay

## Tech used:

- **Python**: Runs the program
- **OpenCV**: Reads your video, draws text/shapes, and saves the final MP4
- **MediaPipe**: Detects human pose points so we can find player heads and place overlays
- **NumPy**: Fast number crunching for coordinates and frame math
- **JSON**: Your timeline of shots/feedback that we overlay on the video

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1) Clone your repo
```bash
git clone https://github.com/yourusername/ai-pickleball-analysis.git
cd ai-pickleball-analysis
```

2) Install deps
```bash
pip install -r requirements.txt
```

### Run (TL;DR)

1) Put your files next to the script:
- your video: `your_video.mp4`
- your analysis: `your_analysis.json` (see schema below)

2) Open `pickleball.py` and set one config:
```python
CONFIGS = {
    1: {
        'json_file': 'your_analysis.json',
        'video_file': 'your_video.mp4',
        'output_file': 'your_output.mp4'
    }
}
CURRENT_CONFIG = 1
```

3) Run it
```bash
python pickleball.py
```

## How the JSON is generated (Google AI Studio)

I used Google AI Studio to create the JSON with timestamps and feedback. This makes the current version not real-time. If you connect the app to the Google API and send frames or scene summaries while playing, you could make it real-time.

Exact prompt I used (paste this and upload your video):

```text
This is me and 3 other people playing pickleball.
Tell me the score throughout the game and who won each point. Tell me what shots each player made and missed.
Tell me what each player did good and what they need to work on. Give me feedback like you're Ben Johns. make the feedback much shorter and more direct, use simple english. I need it in a format similar to this of json and include timestamps, help me do this properly: {
"shots": [
{
"timestamp_of_outcome": "0:07.5",
"result": "missed",
"shot_type": "Jump shot (around free-throw line)",
"total_shots_made_so_far": 0,
"total_shots_missed_so_far": 1,
"total_layups_made_so_far": 0,
"feedback": "You're pushing that ball, not shooting it; get your elbow under, extend fully, and follow through."
},
{
"timestamp_of_outcome": "0:13.0",
"result": "made",
"shot_type": "Three-pointer",
"total_shots_made_so_far": 1,
"total_shots_missed_so_far": 1,
"total_layups_made_so_far": 0,
"feedback": "It went in, but watch that slight fade keep your shoulders square to the hoop through the whole motion."
},
{
"timestamp_of_outcome": "0:21.5",
"result": "made",
"shot_type": "Layup",
"total_shots_made_so_far": 2,
"total_shots_missed_so_far": 1,
"total_layups_made_so_far": 1,
"feedback": "Drive that knee on the layup, protect the ball higher with your off-hand, and finish decisively."
},
{
"timestamp_of_outcome": "0:28.5",
"result": "made",
"shot_type": "Jump shot (free-throw line)",
"total_shots_made_so_far": 3,
"total_shots_missed_so_far": 1,
"total_layups_made_so_far": 1,
"feedback": "Better balance, but that shot pocket and release point must be identical every single time for real consistency."
}
]
}
```

Note: the script expects the schema shown below in “JSON Data Format”. If your AI output differs, adjust the keys or map them before running.

### Map AI output to the expected schema (quick Python shim)

Use this if your AI JSON uses fields like `player`/`result` instead of the exact keys the script expects.

```python
import json

# Input: AI Studio output
with open('ai_output.json', 'r') as f:
    ai = json.load(f)

normalized_shots = []

# Optional score inference if your AI doesn't return current_score
team1_score = 0
team2_score = 0
serving_team = 1

for e in ai.get('shots', []):
    ts = e.get('timestamp_of_outcome') or e.get('timestamp') or '0:00'
    player = e.get('shot_by_player') or e.get('player') or e.get('shooter') or 'Unknown'
    shot_type = e.get('shot_type') or 'Unknown'

    # Prefer explicit winner from AI; otherwise infer from generic result words
    winner = e.get('point_winner')
    if not winner:
        result = (e.get('result') or '').lower()
        if result in {'made', 'win', 'winner', 'ace'}:
            winner = 'Foreground Team'
            team1_score += 1
        elif result in {'miss', 'missed', 'error', 'lose', 'lost'}:
            winner = 'Far Side Team'
            team2_score += 1
        else:
            winner = 'Foreground Team'  # default if unknown

    current_score = e.get('current_score') or f"{team1_score}-{team2_score}-{serving_team}"

    normalized_shots.append({
        'timestamp_of_outcome': ts.split('.')[0],  # trim milliseconds if present
        'shot_by_player': player,
        'shot_type': shot_type,
        'point_winner': winner,
        'current_score': current_score,
        'feedback': e.get('feedback', '')
    })

with open('pickleball.json', 'w') as f:
    json.dump({'shots': normalized_shots}, f, indent=2)

print('Wrote pickleball.json')
```

## JSON Data Format

The magic happens in the JSON analysis file. Here's the required format:

```json
{
  "shots": [
    {
      "timestamp_of_outcome": "0:15",
      "shot_by_player": "Player 1",
      "shot_type": "Forehand Drive",
      "point_winner": "Foreground Team",
      "current_score": "1-0-1",
      "feedback": "Great shot placement! The forehand was executed with perfect timing and placement."
    }
  ]
}
```

### Field Descriptions

- `timestamp_of_outcome`: When the shot outcome occurs (format: "minutes:seconds")
- `shot_by_player`: Name of the player making the shot
- `shot_type`: Type of shot (e.g., "Forehand Drive", "Backhand Slice")
- `point_winner`: "Foreground Team" or "Far Side Team"
- `current_score`: Score in format "team1-team2-serving_team"
- `feedback`: Detailed analysis text to display

## Project Structure

```
ai-pickleball-analysis/
├── pickleball.py          # Main analysis script
├── ball.py               # Alternative ball tracking script
├── requirements.txt      # Python dependencies
├── setup.py             # Automated setup script
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── *.json              # Analysis data files
└── sample_videos/      # Example video files (not in repo)
```

## Configuration Options

### Video Processing Settings

- `process_every_n_frames`: Controls processing frequency (default: every 3rd frame)
- `animation_duration`: Duration of point outcome animations (default: 1.5 seconds)
- `feedback_display_duration`: How long to show feedback text (default: 5 seconds)

### Display Customization

- Player name and arrow styling
- Score display position and colors
- Text font sizes and positioning
- Animation colors and timing

## Tips

- If the video won’t open, double-check the file name and location
- For faster processing, use a smaller-resolution video or increase `process_every_n_frames`

## Performance Tips

- **Video Resolution**: Higher resolution videos provide better pose detection but slower processing
- **Processing Rate**: Adjust `process_every_n_frames` based on your hardware
- **Memory Usage**: Large videos are processed in memory - consider streaming for very long videos

## Troubleshooting

### Common Issues

1. **"Could not open video file"**
   - Check file path and format
   - Ensure video codec is supported by OpenCV

2. **"Error loading JSON file"**
   - Verify JSON syntax is valid
   - Check file path and permissions

3. **Poor pose detection**
   - Ensure good lighting in video
   - Check that players are clearly visible
   - Consider higher resolution source video

### Performance Issues

- Reduce video resolution for faster processing
- Increase `process_every_n_frames` value
- Ensure sufficient RAM for video processing

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Original concept inspired by viral sports analysis demos
- Built with MediaPipe for pose detection
- OpenCV for video processing and graphics

## Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Search existing GitHub issues
3. Create a new issue with detailed information

---

**Made with ❤️ for sports analysis and computer vision enthusiasts**