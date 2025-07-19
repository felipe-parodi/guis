<!-- plan_qc_bbox_editor.md -->

Want a video tracking QC GUI either local or in browser


Features:
- Efficient video streaming without full memory caching
- Good UI structure w timeline, playback controls, and detection markers
- Handles different JSON formats for bbox data
- Proper threading for video decoding
- Downsampling option for performance
- Smart tracking options
  - Hybrid: OpenCV's built-in trackers for short-term tracking + simple interpolation + template matching for ReID
    - Interpolation (My #1 Recommendation): This is the most practical and effective solution for this use case. The workflow is simple for the user:
        Go to a frame where the track exists (Frame A).
        Go to a later frame where the track reappears (Frame B).
        Create/correct the box for the same track ID on Frame B.
        Press an "Interpolate" hotkey.
        The tool then automatically creates bounding boxes for all frames between A and B by linearly interpolating the corner coordinates. This is extremely fast and perfectly solves the "missed detection" problem.
        Short-Term Tracker (Also a great option): Use one of OpenCV's built-in tracker objects (e.g., cv2.TrackerCSRT_create). The workflow:
        User draws a box on the current frame.
        They press a "Track Forward" button.
        The tool initializes the tracker with that box and runs it for the next N frames, creating new bboxes automatically. This is more "intelligent" than interpolation but can still drift over time. It's excellent for filling short gaps.
- Bbox editing: bbox drag/resize, delete fxnality
- Conf-threshold filtering with slider to dynamically hide bbox below certain conf score
- Add bbox with click-drag
- Batch operations: interpolate gaps, merge/split tracks
- Undo/redo system
- Double-buffered pixmaps when scrubbing?
- On play, skip drawing text labels if FPS>15 to avoid Qt bottleneck; draw once when paused
- Multiple bbox selection
- Copy/paste bboxes across frames
- Keyboard shortcuts for all operations
- GPU acceleration for tracking if available
- Multi-threaded batch processing? Eh
- Export to diff formats; Export “diff” JSON rather than full overwrite.
- Visualization options (trajectory paths, heatmaps)
- Auto-save
- Hotkeys: Q/E for prev/next detection, WASD for bbox nudging
- Track history to show where track has been in prev frames
- Validation: highlight suspicious tracks (sudden jumps, ID switches)

Bugs
- Track ID assignment