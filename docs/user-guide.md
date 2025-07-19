# Video Bbox-Track Editor - User Guide

A comprehensive tool for rapid bbox editing and quality control for video tracking data.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Operations](#basic-operations)
3. [Bbox Editing](#bbox-editing)
4. [Advanced Features](#advanced-features)
5. [Keyboard Shortcuts](#keyboard-shortcuts)
6. [Export Options](#export-options)
7. [Tips & Troubleshooting](#tips--troubleshooting)

## Getting Started

### Installation

**Using conda:**
```bash
cd guis/
conda env create -f environment.yml
conda activate guis
```

**Using uv:**
```bash
cd guis/
uv sync
```

### First Run

1. **Prepare your data:**
   - Create a CSV file with video and annotation paths:
     ```csv
     video_path,bbox_json_path
     /path/to/video1.mp4,/path/to/bbox1.json
     /path/to/video2.mp4,/path/to/bbox2.json
     ```

2. **Launch the application:**
   ```bash
   python bbox_track_gui.py --csv your_data.csv
   ```

3. **Load your first video:**
   - Select a video from the dropdown menu
   - The video will load automatically with any existing annotations

## Basic Operations

### Video Playback

- **Play/Pause**: Click the Play button or press `Space`
- **Frame Navigation**: 
  - Use arrow keys (`←`/`→`) for single frame steps
  - Drag the timeline slider for quick navigation
  - Click directly on the timeline to jump to a specific frame

### Speed Control

- **Change Playback Speed**: Use the speed dropdown (0.25x to 2x)
- **Real-time Navigation**: The timeline updates in real-time during playback

### Video Display

- **Zoom**: Mouse wheel to zoom in/out
- **Pan**: Click and drag to move around when zoomed
- **Fit to Window**: The video automatically scales to fit the display area

## Bbox Editing

### Adding Bboxes

1. **Enter Add Mode**: Click "Add Bbox" button or press `A`
2. **Draw Bbox**: Click and drag on the video to create a new bounding box
3. **Assign Track ID**: Press number keys `1-9` to assign a track ID
4. **Exit Add Mode**: Click "Add Bbox" again or press `A`

### Editing Existing Bboxes

- **Select**: Click on any bbox to select it (highlighted in different color)
- **Move**: Drag the selected bbox to reposition
- **Resize**: Drag the corner/edge handles to resize
- **Delete**: Select a bbox and press `Delete`

### Track ID Management

- **Assign Track ID**: Select a bbox and press `1-9` keys
- **Visual Indicators**: Each track ID has a unique color
- **Track Persistence**: Track IDs persist across frames for continuity

## Advanced Features

### Smart Interpolation

**Purpose**: Automatically fill in missing frames between keyframes with smooth motion.

**How to use:**
1. Ensure the same track ID exists on two different frames
2. Select a bbox with the desired track ID
3. Click "Interpolate" button or press `I`
4. The system fills intermediate frames with smoothstep motion estimation

**Features:**
- **Smoothstep Motion**: Natural acceleration/deceleration
- **Size Interpolation**: Gradual size changes between keyframes
- **Confidence Assignment**: Interpolated boxes get 0.99 confidence

### Confidence Filtering

**Purpose**: Filter and visualize detections based on confidence scores.

**Controls:**
- **Confidence Slider**: Adjust the minimum confidence threshold (0.00-1.00)
- **Visual Feedback**: Low-confidence boxes shown with reduced opacity (0.3)
- **Real-time Updates**: Filter updates immediately as you adjust the slider

### Short-term Tracking

**Purpose**: Automatically track selected objects forward through frames.

**Available Trackers:**
- **CSRT**: Most accurate, good for complex scenarios
- **KCF**: Fast, good for translation and scale changes  
- **MOSSE**: Very fast, good for simple tracking
- **MIL**: Good for partial occlusions

**How to use:**
1. Select a bbox you want to track
2. Choose tracker type from dropdown
3. Click "Track Forward" or press `T`
4. Click "Stop Tracking" to end automatic tracking

### Visualization Options

**Track Trails:**
- **Enable**: Check "Show Track Trails"
- **Purpose**: Shows the path history of each track
- **Visual**: Fading trail effect showing recent positions

**Trajectories:**
- **Enable**: Check "Show Trajectories" 
- **Purpose**: Display full trajectory paths for all tracks
- **Use Case**: Analyze movement patterns and detect anomalies

**Auto-save:**
- **Enable**: Check "Auto-save"
- **Frequency**: Configurable timer (saves periodically)
- **Safety**: Prevents data loss during long editing sessions

### Quality Control Features

**Detection Navigation:**
- **Next Detection**: Press `E` to jump to next frame with detections
- **Previous Detection**: Press `Q` to jump to previous frame with detections
- **Efficiency**: Quickly review only frames that need attention

**Label Display Control:**
- **FPS-based**: Labels automatically hide during fast playback for performance
- **Threshold**: Configurable FPS threshold for label visibility
- **Manual Override**: Force show/hide labels regardless of FPS

## Keyboard Shortcuts

### Essential Controls
| Key | Action |
|-----|--------|
| `Space` | Play/Pause video |
| `←` / `→` | Previous/Next frame |
| `A` | Toggle Add Bbox mode |
| `Delete` | Delete selected bbox |
| `Ctrl+S` | Save changes |

### Track Management
| Key | Action |
|-----|--------|
| `1-9` | Assign track ID to selected bbox |
| `I` | Interpolate selected track |
| `T` | Start tracking selected bbox forward |

### Navigation
| Key | Action |
|-----|--------|
| `Q` | Jump to previous detection frame |
| `E` | Jump to next detection frame |

### File Operations
| Key | Action |
|-----|--------|
| `Ctrl+O` | Open file dialog |
| `Ctrl+E` | Export dialog |

## Export Options

### COCO Format

**Purpose**: Standard format for computer vision datasets.

**Features:**
- Complete video metadata (width, height, frame count)
- Proper category and annotation structure
- Compatible with most training frameworks

**Output**: JSON file with COCO-compliant structure

### YOLO Format

**Purpose**: Format for YOLO model training.

**Features:**
- Normalized coordinates (0-1 range)
- One text file per frame
- Class ID mapping

**Output**: Directory with `.txt` files for each frame

### CSV Format

**Purpose**: Tabular data for analysis and processing.

**Columns:**
- `video_path`: Source video file
- `frame_id`: Frame number
- `track_id`: Object track identifier
- `x1, y1, x2, y2`: Bounding box coordinates
- `confidence`: Detection confidence score

**Use Cases**: Data analysis, statistics, custom processing

## Tips & Troubleshooting

### Performance Optimization

**For Large Videos:**
- Use downsampling option to reduce memory usage
- Enable auto-save to prevent data loss
- Close other applications to free up memory

**For Smooth Playback:**
- Lower playback speed for complex scenes
- Use frame-by-frame navigation for precise editing
- Disable trails/trajectories if performance is slow

### Best Practices

**Efficient Workflow:**
1. **Quick Review**: Use `Q`/`E` to jump between detection frames
2. **Bulk Editing**: Use interpolation for long sequences
3. **Quality Control**: Use confidence filtering to focus on uncertain detections
4. **Regular Saves**: Enable auto-save or use `Ctrl+S` frequently

**Track ID Management:**
- Use consistent track IDs across the entire video
- Start with track ID 1 and increment sequentially
- Use interpolation to maintain track continuity

### Common Issues

**"Bboxes not displaying":**
- Check that your JSON format is supported (both nested `[[x,y,x,y]]` and flat `[x,y,x,y]` formats work)
- Verify confidence threshold isn't filtering out all detections
- Ensure bbox coordinates are within video boundaries

**"Confidence slider moves video":**
- This indicates a signal connection issue - restart the application
- Check that you're using the confidence slider (top slider) not the timeline (bottom slider)

**"GUI crashes when adding bbox":**
- Ensure you have sufficient memory available
- Try reducing video resolution with downsampling option
- Check that OpenCV is properly installed with tracking modules

**"Tracker not available":**
- Install opencv-contrib-python: `pip install opencv-contrib-python`
- Some OpenCV versions have different tracker APIs - the tool automatically detects and adapts

### Data Format Support

**JSON Format Examples:**

Nested format:
```json
{
  "frame_id": 100,
  "instances": [
    {
      "bbox": [[x1, y1, x2, y2]],
      "track_id": 1,
      "confidence": 0.95
    }
  ]
}
```

Flat format:
```json
{
  "frame_id": 100,
  "instances": [
    {
      "bbox": [x1, y1, x2, y2],
      "track_id": 1,
      "confidence": 0.95
    }
  ]
}
```

Both formats are automatically detected and supported.

### Getting Help

**Debug Information:**
- Check the console output for error messages
- Look for DEBUG messages that show what the application is doing
- Note any specific error traces for troubleshooting

**Performance Monitoring:**
- Monitor memory usage during long sessions
- Check if auto-save is working properly
- Verify export operations complete successfully

---

## Quick Reference Card

**Most Used Functions:**
- `Space`: Play/Pause
- `A`: Add bbox mode
- `1-9`: Set track ID
- `I`: Interpolate
- `Q/E`: Navigate detections
- `Ctrl+S`: Save

**Remember**: The tool auto-saves your work when enabled, but manual saves with `Ctrl+S` are always recommended for important edits.