# Contributing to AI Sports Analysis Tool

Thank you for your interest in contributing! This document provides detailed information about the project structure and how to contribute effectively.

## Project Architecture

### Core Components

#### `pickleball.py` - Main Analysis Engine
- **Purpose**: Primary video analysis and overlay generation
- **Key Functions**:
  - `parse_timestamp()`: Converts time strings to seconds
  - `timestamp_to_frame()`: Maps timestamps to video frames
  - `wrap_text()`: Text wrapping for overlay captions
  - `get_animation_color()`: Color animation for score changes

#### `ball.py` - Alternative Ball Tracking
- **Purpose**: Specialized ball tracking implementation
- **Usage**: For ball-specific analysis vs player tracking

### Configuration System

The `CONFIGS` dictionary allows multiple video/analysis combinations:

```python
CONFIGS = {
    1: {
        'json_file': 'analysis1.json',
        'video_file': 'video1.mp4', 
        'output_file': 'output1.mp4'
    }
}
```

### Data Flow

1. **Input**: Video file + JSON analysis data
2. **Processing**: MediaPipe pose detection every N frames
3. **Overlay**: Score tracking, shot info, feedback text
4. **Output**: Processed video with analysis overlay

### JSON Schema

#### Required Fields
- `timestamp_of_outcome`: String in "MM:SS" format
- `shot_by_player`: Player name string
- `shot_type`: Shot description string  
- `point_winner`: "Foreground Team" | "Far Side Team"
- `current_score`: Score string (format: "team1-team2-serving")
- `feedback`: Analysis text for display

#### Legacy Support
Also supports older schema with:
- `concluding_shot_player` instead of `shot_by_player`
- `concluding_shot_type` instead of `shot_type`
- `score_after_point` instead of `current_score`

## Development Setup

### Prerequisites
- Python 3.8+
- OpenCV with video codec support
- Git for version control

### Local Development
```bash
# Clone your fork
git clone https://github.com/yourusername/ai-sports-analysis.git
cd ai-sports-analysis

# Install in development mode
pip install -e .

# Install development dependencies
pip install -r requirements-dev.txt  # If you create this

# Run tests
python -m pytest  # If you add tests
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings for functions
- Keep functions focused and modular

### Performance Considerations

#### Video Processing
- Default processes every 3rd frame (`process_every_n_frames`)
- MediaPipe pose detection is the bottleneck
- Balance accuracy vs speed based on use case

#### Memory Management
- All frames stored in `processed_frames` list
- Consider streaming for very long videos
- Monitor memory usage with large files

#### Display Optimization
- Use separate processing and display video streams
- Scale coordinates between resolutions
- Optimize text rendering for performance

## Adding Features

### New Analysis Types
1. Create new JSON schema for your analysis type
2. Add parsing logic in main processing loop
3. Implement overlay rendering
4. Update configuration system

### New Overlay Elements
1. Define drawing functions
2. Add position calculations
3. Implement animation if needed
4. Update display loop

### Video Format Support
1. Test with OpenCV video support
2. Add format validation
3. Update documentation
4. Consider codec compatibility

## Testing

### Test Videos
- Use short clips for faster iteration
- Test different resolutions and framerates
- Verify pose detection quality
- Check overlay positioning

### Analysis Data
- Validate JSON schema compliance
- Test edge cases (missing data, invalid timestamps)
- Verify score calculation logic
- Test feedback text wrapping

## Submitting Changes

### Pull Request Process
1. Fork the repository
2. Create feature branch from main
3. Make your changes with tests
4. Update documentation as needed
5. Submit PR with clear description

### PR Requirements
- Clear description of changes
- Any breaking changes noted
- Updated documentation
- Performance impact assessment

### Commit Guidelines
- Use conventional commit format
- Be specific about changes made
- Reference issues if applicable

## Future Roadmap

### Real-time Implementation
- Live camera feed processing
- Reduced latency pipeline
- Mobile app adaptation

### AI Integration
- Direct API integration with analysis services
- Automated shot classification
- Real-time coaching feedback

### Enhanced Visualizations
- 3D pose visualization
- Heat maps and statistics
- Advanced graphics overlays

## Getting Help

- Check existing issues before creating new ones
- Provide video samples and JSON data when reporting bugs
- Include system information (OS, Python version, etc.)
- Be specific about error messages and steps to reproduce

## Code Review Guidelines

### What We Look For
- Code clarity and maintainability
- Performance impact assessment
- Backward compatibility
- Documentation updates
- Test coverage

### Common Issues
- Hard-coded values instead of configuration
- Memory leaks in video processing
- Missing error handling
- Inconsistent naming conventions

---

Ready to contribute? Start by exploring the codebase and trying different video analysis scenarios!
