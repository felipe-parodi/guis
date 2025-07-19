# Video Tracking Quality Control GUI

A fast, interactive video player for reviewing and correcting object tracking predictions. Features smart interpolation, confidence filtering, track trails, auto-save, and export capabilities.

## Features

### Core Functionality
- **Smart Interpolation** with smoothstep motion estimation for natural tracking
- **Confidence Filtering** with reduced opacity (0.3) for low-confidence detections
- **Track Trails** visualization showing object history across frames
- **Auto-save** functionality with configurable timer
- **Q/E Navigation** for jumping between detection frames
- **Short-term Tracking** using OpenCV trackers (CSRT, KCF, MOSSE, MIL)

### Editing Tools
- Interactive bbox editing with drag/resize handles
- Add new bboxes with click-drag
- Delete selected bboxes
- Track ID assignment (1-9 keys)
- Confidence threshold filtering

### Visualization Options
- Track trajectory paths with fading effects
- Heatmap generation for movement analysis
- Suspicious track highlighting for quality control
- Configurable label drawing based on FPS threshold

### Export Formats
- **COCO** format with proper video metadata
- **YOLO** format for training datasets
- **CSV** format for data analysis

## Quick Start

### Installation

Using conda:
```bash
conda env create -f environment.yml
conda activate guis
```

Using uv:
```bash
uv sync
```

### Usage

```bash
python bbox_track_gui.py --csv tracking_data.csv
```

CSV format:
```csv
video_path,bbox_json_path
/path/to/video1.mp4,/path/to/bbox1.json
/path/to/video2.mp4,/path/to/bbox2.json
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| Left/Right Arrow | Step one frame |
| 1-9 | Assign track ID to selected box |
| A | Enter 'Add Bbox' mode |
| Delete | Delete selected bbox |
| Ctrl+S | Save changes |
| I | Interpolate selected track |
| Q/E | Jump to prev/next detection |
| T | Start tracking selected box forward |

## Architecture

### Modular Design
- `utils/bbox_utils.py` - BBox and BBoxManager classes with smart interpolation
- `utils/tracker.py` - OpenCV short-term tracking with multiple algorithm support
- `utils/visualization.py` - Trajectory paths, heatmaps, and visualization overlays
- `utils/export_utils.py` - Export functionality for COCO/YOLO/CSV formats

### Performance Features
- Frame caching for smooth video scrubbing
- Double-buffering for responsive UI
- Conditional label rendering based on FPS
- Efficient video streaming without full memory caching

## Configuration

### Environment Variables
- `OPENCV_FFMPEG_READ_ATTEMPTS` - Number of read attempts for video files
- `OPENCV_FFMPEG_CAPTURE_OPTIONS` - Threading options for video capture

### Tracker Types
- **CSRT** - Accurate but slower, good for complex scenarios
- **KCF** - Fast, good for translation and scale changes
- **MOSSE** - Very fast but less accurate, good for simple tracking
- **MIL** - Good for partial occlusions and robustness

## JSON Format Support

The tool automatically parses various JSON annotation formats:
- List of frame objects with `instances` arrays
- Dictionary with `frames` or `instance_info` keys
- Frame-indexed dictionaries with numeric keys

Example format:
```json
{
  "frames": [
    {
      "frame_id": 0,
      "instances": [
        {
          "bbox": [x1, y1, x2, y2],
          "track_id": 1,
          "confidence": 0.95
        }
      ]
    }
  ]
}
```

## Development

### Dependencies
- PyQt5 >= 5.15.0
- opencv-contrib-python >= 4.5.0 (includes tracking modules)
- numpy >= 1.20.0
- pandas >= 1.3.0

### Code Quality
The codebase follows modular architecture principles with:
- Separation of concerns between UI, data management, and algorithms
- Comprehensive error handling and user feedback
- Extensive documentation and type hints
- Performance optimizations for real-time video processing

### Testing
Test with various video formats and annotation structures. The tool is designed to handle:
- Different video codecs and resolutions
- Long videos with efficient streaming
- Large annotation datasets
- Various OpenCV versions and tracker availability
