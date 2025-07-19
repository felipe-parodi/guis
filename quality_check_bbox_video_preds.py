#!/usr/bin/env python3
"""
Video Tracking Quality Control GUI

A fast, streaming video player for reviewing and correcting object tracking predictions.
Handles long videos (40-120 minutes) efficiently without caching to memory.

Usage:
    python video_tracking_qc.py --csv tracking_data.csv
    
CSV Format:
    video_path,bbox_json_path
    /path/to/video1.mp4,/path/to/bbox1.json
    /path/to/video2.mp4,/path/to/bbox2.json

Requirements:
    pip install PyQt5 opencv-python-headless numpy pandas

Note: If you encounter FFmpeg errors with certain videos, try:
    1. pip install opencv-python==4.5.5.64
    2. Convert problematic videos using:
       ffmpeg -i input.mp4 -c:v libx264 -preset fast -crf 22 output.mp4
    3. The app now runs FFmpeg in single-threaded mode to avoid threading issues
"""

import sys
import os

# Set environment variable before importing cv2 to avoid threading issues
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"

import json
import csv
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
import queue
import time

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


@dataclass
class BBox:
    """Bounding box with track ID"""
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    selected: bool = False


class VideoDecoder(QThread):
    """Efficient video decoder thread"""
    frameReady = pyqtSignal(np.ndarray, int)
    positionChanged = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.cap = None
        self.video_path = None
        self.fps = 30.0
        self.total_frames = 0
        self.current_frame = 0
        self.playing = False
        self.playback_speed = 1.0
        self.seek_frame = -1
        self.seek_lock = threading.Lock()
        self.downsampling = False
        self.target_size = (960, 540)  # Downsampled size
        self.force_refresh = False
        self.is_loading = False  # Prevent decoding during video load
        self.should_stop = False  # Clean shutdown flag
        
    def load_video(self, video_path: str):
        """Load video and extract metadata"""
        self.is_loading = True
        self.playing = False  # Stop playback
        
        # Wait a bit for any pending operations
        self.msleep(100)
        
        # Release previous video with lock
        with self.seek_lock:
            if self.cap:
                print(f"Releasing previous video...")
                self.cap.release()
                self.cap = None
                self.msleep(50)  # Give FFmpeg time to cleanup
                print(f"Previous video released")
            
            self.video_path = video_path
            
            # Try to open video with error handling
            try:
                # For problematic videos, try disabling threading
                self.cap = cv2.VideoCapture(video_path)
                
                # Try to set single-threaded decoding to avoid FFmpeg threading issues
                if hasattr(cv2, 'CAP_PROP_THREAD_COUNT'):
                    self.cap.set(cv2.CAP_PROP_THREAD_COUNT, 1)
                
                if not self.cap.isOpened():
                    # Try with different backend
                    self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                    if not self.cap.isOpened():
                        raise ValueError(f"Cannot open video: {video_path}")
                    
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.current_frame = 0

                # If frame count is invalid, try a more robust method
                if self.total_frames <= 0:
                    print(f"Warning: Could not get frame count for {video_path}")
                    self.total_frames = 1000000  # Set a large default
                
                # Read first frame to get dimensions
                ret, frame = self.cap.read()
                if ret:
                    self.original_size = (frame.shape[1], frame.shape[0])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    print(f"Video loaded successfully: {self.original_size[0]}x{self.original_size[1]}, {self.total_frames} frames @ {self.fps} fps")
                else:
                    raise ValueError(f"Cannot read first frame from video: {video_path}")
                    
            except Exception as e:
                self.is_loading = False
                raise
                
        self.is_loading = False
        return self.fps, self.total_frames
        
    def seek_to_frame(self, frame_num: int):
        """Seek to specific frame"""
        if self.is_loading:
            return  # Don't seek while loading
        with self.seek_lock:
            if self.cap and self.cap.isOpened():
                self.seek_frame = max(0, min(frame_num, self.total_frames - 1))
                self.force_refresh = True  # Force a frame refresh after seek
            
    def set_downsampling(self, enabled: bool):
        """Toggle downsampling for performance"""
        self.downsampling = enabled
        
    def run(self):
        """Main decode loop"""
        frame_time = 1.0 / 30.0  # Default frame time
        last_frame_time = time.time()
        
        while not self.should_stop:
            # Skip processing if loading new video
            if self.is_loading:
                self.msleep(50)
                continue
                
            if not self.cap or not self.cap.isOpened():
                self.msleep(100)
                continue
                
            # Update frame time based on current settings
            if self.fps > 0:
                frame_time = 1.0 / self.fps / self.playback_speed
                
            seeked = False
            # Handle seeking
            with self.seek_lock:
                if self.seek_frame >= 0 and self.cap:
                    try:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                        self.current_frame = self.seek_frame
                        self.seek_frame = -1
                        last_frame_time = time.time()
                        seeked = True
                    except Exception as e:
                        print(f"Seek error: {e}")
                        self.seek_frame = -1
            
            # Decode frame if playing, seeking, or force refresh
            if (self.playing or seeked or self.force_refresh) and not self.is_loading:
                current_time = time.time()
                
                # Wait for appropriate frame time if playing
                if self.playing and not seeked and not self.force_refresh:
                    elapsed = current_time - last_frame_time
                    if elapsed < frame_time:
                        self.msleep(int((frame_time - elapsed) * 1000))
                        continue
                    
                try:
                    ret, frame = self.cap.read()
                    if ret:
                        # Downsample if enabled
                        if self.downsampling:
                            frame = cv2.resize(frame, self.target_size)
                            
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        self.frameReady.emit(frame, self.current_frame)
                        self.positionChanged.emit(self.current_frame)
                        
                        if self.playing:
                            self.current_frame += 1
                        last_frame_time = time.time()
                        self.force_refresh = False
                    else:
                        # End of video or read error
                        if self.playing:
                            self.playing = False
                            print("End of video reached or read error")
                except Exception as e:
                    print(f"Frame decode error: {e}")
                    self.playing = False
                    
            else:
                self.msleep(30)  # Sleep when not playing
                
    def stop_thread(self):
        """Stop the decoder thread cleanly"""
        self.should_stop = True
        self.playing = False
        if self.cap:
            with self.seek_lock:
                self.cap.release()
                self.cap = None
                
    def play(self):
        """Start playback"""
        self.playing = True
        
    def pause(self):
        """Pause playback"""
        self.playing = False
        
    def set_speed(self, speed: float):
        """Set playback speed"""
        self.playback_speed = speed


class BBoxManager:
    """Manages bounding box data and track IDs"""
    
    def __init__(self):
        self.bbox_data = {}  # frame_id -> List[BBox]
        self.json_path = None
        self.json_data = None
        self.raw_json_data = None  # Store original structure
        self.modified = False
        self.active_track_ids = set()
        
    def load_json(self, json_path: str):
        """Load bbox JSON data"""
        self.json_path = json_path
        print(f"Loading JSON from: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                self.raw_json_data = json.load(f)
                
            # Parse JSON into bbox_data
            self.bbox_data = {}

            # Check if data is wrapped in 'instance_info' or similar
            if isinstance(self.raw_json_data, dict):
                # Try common wrapper keys
                for key in ['instance_info', 'frames', 'results', 'data']:
                    if key in self.raw_json_data:
                        self.json_data = self.raw_json_data[key]
                        print(f"Found frame data under key: {key}")
                        break
                else:
                    # Maybe the dict uses frame numbers as keys
                    print("JSON is a dict, trying to parse frame numbers as keys...")
                    self.json_data = []
                    for frame_key, frame_data in self.raw_json_data.items():
                        if frame_key.isdigit() and isinstance(frame_data, dict):
                            frame_data['frame_id'] = int(frame_key)
                            self.json_data.append(frame_data)
            else:
                # Assume it's a list of frame objects
                self.json_data = self.raw_json_data
                
            if not self.json_data:
                print("Warning: No frame data found in JSON")
                return
                
            print(f"Found {len(self.json_data)} frames in JSON")
            
            # Parse frames
            for i, frame_obj in enumerate(self.json_data):
                try:
                    # Get frame ID
                    frame_id = frame_obj.get('frame_id', frame_obj.get('frame', i))
                    
                    bboxes = []
                    instances = frame_obj.get('instances', [])
                    
                    if not isinstance(instances, list):
                        continue
                        
                    for instance in instances:
                        # Handle different bbox formats
                        bbox_data = instance.get('bbox', [])
                        if not bbox_data:
                            continue
                            
                        # Check if bbox is nested
                        if isinstance(bbox_data[0], list):
                            bbox_coords = bbox_data[0]
                        else:
                            bbox_coords = bbox_data
                            
                        if len(bbox_coords) < 4:
                            continue
                        
                        # Extract track_id if present
                        track_id = None
                        instance_id_val = instance.get('instance_id', {})
                        if isinstance(instance_id_val, (int, str)) and str(instance_id_val).isdigit():
                            track_id = int(instance_id_val)
                        elif isinstance(instance_id_val, list) and instance_id_val and isinstance(instance_id_val[0], (int, str)) and str(instance_id_val[0]).isdigit():
                            track_id = int(instance_id_val[0])
                        elif isinstance(instance_id_val, dict):
                            # Look for common keys for track ID
                            track_id = instance_id_val.get('track_id')
                            if track_id is None:
                                for key in ['value', 'id', 'data']:
                                    if key in instance_id_val and isinstance(instance_id_val[key], (int, str)) and str(instance_id_val[key]).isdigit():
                                        track_id = int(instance_id_val[key])
                                        break

                        # Extract confidence score
                        confidence = None
                        bbox_score_val = instance.get('bbox_score', instance.get('score'))
                        if isinstance(bbox_score_val, (int, float)):
                            confidence = float(bbox_score_val)
                        elif isinstance(bbox_score_val, list) and bbox_score_val and isinstance(bbox_score_val[0], (int, float)):
                            confidence = float(bbox_score_val[0])
                        elif isinstance(bbox_score_val, dict):
                            # Try to find a numeric value within the dictionary
                            for key in ['value', 'score', 'data']:
                                if key in bbox_score_val and isinstance(bbox_score_val[key], (int, float)):
                                    confidence = float(bbox_score_val[key])
                                    break
                             
                        bbox = BBox(
                            x1=float(bbox_coords[0]),
                            y1=float(bbox_coords[1]),
                            x2=float(bbox_coords[2]),
                            y2=float(bbox_coords[3]),
                            track_id=track_id,
                            confidence=confidence
                        )
                        bboxes.append(bbox)
                        
                        if track_id is not None:
                            self.active_track_ids.add(track_id)
                            
                    self.bbox_data[frame_id] = bboxes
                    
                except Exception as e:
                    print(f"Error parsing frame {i}: {e}")
                    continue
                    
            print(f"Successfully parsed {len(self.bbox_data)} frames with detections")
            
        except Exception as e:
            print(f"Error loading JSON: {e}")
            raise
            
    def get_bboxes(self, frame_id: int) -> List[BBox]:
        """Get bboxes for a specific frame"""
        return self.bbox_data.get(frame_id, [])
        
    def set_track_id(self, frame_id: int, bbox_idx: int, track_id: int):
        """Set track ID for a specific bbox"""
        if frame_id in self.bbox_data and bbox_idx < len(self.bbox_data[frame_id]):
            self.bbox_data[frame_id][bbox_idx].track_id = track_id
            self.active_track_ids.add(track_id)
            self.modified = True
            
            # Update JSON data - handle wrapped structure
            if isinstance(self.raw_json_data, dict) and 'instance_info' in self.raw_json_data:
                frame_list = self.raw_json_data['instance_info']
            else:
                frame_list = self.json_data
                
            for frame_obj in frame_list:
                if frame_obj.get('frame_id', frame_obj.get('frame', -1)) == frame_id:
                    if bbox_idx < len(frame_obj.get('instances', [])):
                        # Ensure instance_id is a dict
                        if not isinstance(frame_obj['instances'][bbox_idx].get('instance_id'), dict):
                            frame_obj['instances'][bbox_idx]['instance_id'] = {}
                        frame_obj['instances'][bbox_idx]['instance_id']['track_id'] = track_id
                    break
                    
    def save_json(self, save_path: Optional[str] = None):
        """Save modified JSON data"""
        if not self.modified:
            return
            
        path = save_path or self.json_path
        backup_path = path + '.backup'
        
        # Create backup
        if os.path.exists(path):
            os.rename(path, backup_path)
            
        # Save modified data - preserve original structure
        with open(path, 'w') as f:
            json.dump(self.raw_json_data, f, indent=2)
            
        self.modified = False
        print(f"Saved changes to {path}")
        
    def get_frames_with_detections(self) -> List[int]:
        """Get list of frames that have detections"""
        return sorted([fid for fid, bboxes in self.bbox_data.items() if bboxes])


class DetectionTimeline(QSlider):
    """Custom timeline slider that shows detection markers"""
    
    def __init__(self, orientation=Qt.Horizontal):
        super().__init__(orientation)
        self.detection_frames = []
        self.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 10px;
                background: #333;
                border-radius: 5px;
            }
            QSlider::handle:horizontal {
                background: #fff;
                border: 2px solid #999;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -4px 0;
            }
            QSlider::handle:horizontal:hover {
                background: #ddd;
            }
        """)
        
    def set_detection_frames(self, frames: List[int]):
        """Set frames that have detections"""
        self.detection_frames = frames
        self.update()
        
    def paintEvent(self, event):
        """Custom paint to show detection markers"""
        super().paintEvent(event)
        
        if not self.detection_frames:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw detection markers
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, QStyleOptionSlider(), QStyle.SC_SliderGroove, self
        )
        
        for frame in self.detection_frames:
            if self.minimum() <= frame <= self.maximum():
                # Calculate position
                pos_ratio = (frame - self.minimum()) / max(1, self.maximum() - self.minimum())
                x = groove_rect.left() + pos_ratio * groove_rect.width()
                y = groove_rect.center().y()
                
                # Draw marker
                painter.setPen(QPen(Qt.red, 2))
                painter.drawLine(int(x), int(y - 5), int(x), int(y + 5))
        
        painter.end()


class VideoCanvas(QGraphicsView):
    """Custom canvas for video display and bbox interaction"""
    
    bboxClicked = pyqtSignal(int)  # Emits bbox index when clicked
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        
        self.pixmap_item = None
        self.bbox_items = []
        self.current_frame_array = None  # Store current frame
        self.scale_factor = 1.0
        
    def display_frame(self, frame: np.ndarray):
        """Display video frame"""
        self.current_frame_array = frame  # Store frame
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        if self.pixmap_item:
            self.pixmap_item.setPixmap(pixmap)
        else:
            self.pixmap_item = self.scene.addPixmap(pixmap)
            
        # Fit in view
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        
    def display_bboxes(self, bboxes: List[BBox], scale_x: float = 1.0, scale_y: float = 1.0):
        """Display bounding boxes"""
        # Clear previous boxes
        for item in self.bbox_items:
            self.scene.removeItem(item)
        self.bbox_items.clear()
        
        # Draw new boxes
        for i, bbox in enumerate(bboxes):
            # Scale coordinates if using downsampled video
            x1 = bbox.x1 * scale_x
            y1 = bbox.y1 * scale_y
            x2 = bbox.x2 * scale_x
            y2 = bbox.y2 * scale_y
            
            # Create rectangle
            rect = QGraphicsRectItem(x1, y1, x2-x1, y2-y1)
            
            # Set appearance based on selection and track_id
            if bbox.selected:
                pen = QPen(Qt.yellow, 3)
            elif bbox.track_id is not None:
                # Use different colors for different track IDs
                colors = [Qt.red, Qt.green, Qt.blue, Qt.cyan, Qt.magenta]
                color = colors[bbox.track_id % len(colors)]
                pen = QPen(color, 2)
            else:
                pen = QPen(Qt.white, 2)
                
            rect.setPen(pen)
            rect.setData(0, i)  # Store bbox index
            
            # Add track ID and confidence text
            label_parts = []
            if bbox.track_id is not None:
                label_parts.append(f"ID: {bbox.track_id}")
            if bbox.confidence is not None:
                label_parts.append(f"({bbox.confidence:.2f})")

            if label_parts:
                label_text = " ".join(label_parts)
                text = QGraphicsTextItem(label_text)
                text.setPos(x1, y1 - 20)
                text.setDefaultTextColor(pen.color())
                font = QFont("Arial", 12, QFont.Bold)
                text.setFont(font)
                self.scene.addItem(text)
                self.bbox_items.append(text)
            
            self.scene.addItem(rect)
            self.bbox_items.append(rect)
            
    def mousePressEvent(self, event):
        """Handle mouse clicks on bboxes"""
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            item = self.scene.itemAt(pos, QTransform())
            
            if isinstance(item, QGraphicsRectItem):
                bbox_idx = item.data(0)
                if bbox_idx is not None:
                    self.bboxClicked.emit(bbox_idx)
                    
        super().mousePressEvent(event)


class TrackingQCWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Tracking Quality Control")
        self.setGeometry(100, 100, 1400, 900)
        
        self.video_decoder = VideoDecoder()
        self.bbox_manager = BBoxManager()
        self.current_video_idx = 0
        self.video_list = []
        self.selected_bbox_idx = None
        self.downsampling = False
        
        # Scrubbing optimization
        self.scrub_timer = QTimer()
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.timeout.connect(self.perform_delayed_seek)
        self.pending_seek_frame = -1
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Setup UI components"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        
        # File selection
        self.file_label = QLabel("No file loaded")
        toolbar.addWidget(self.file_label)
        
        # Navigation between videos
        self.prev_video_btn = QPushButton("← Prev Video")
        self.prev_video_btn.clicked.connect(self.prev_video)
        toolbar.addWidget(self.prev_video_btn)
        
        self.next_video_btn = QPushButton("Next Video →")
        self.next_video_btn.clicked.connect(self.next_video)
        toolbar.addWidget(self.next_video_btn)
        
        toolbar.addStretch()
        
        # Downsampling toggle
        self.downsample_cb = QCheckBox("Downsample (faster)")
        self.downsample_cb.toggled.connect(self.toggle_downsampling)
        toolbar.addWidget(self.downsample_cb)
        
        # Save button
        self.save_btn = QPushButton("Save Changes")
        self.save_btn.clicked.connect(self.save_changes)
        self.save_btn.setEnabled(False)
        toolbar.addWidget(self.save_btn)
        
        layout.addLayout(toolbar)
        
        # Video canvas
        self.video_canvas = VideoCanvas()
        layout.addWidget(self.video_canvas, stretch=1)
        
        # Timeline
        timeline_layout = QVBoxLayout()
        
        # Timeline slider with detection markers
        self.timeline = DetectionTimeline(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.valueChanged.connect(self.seek_to_frame)
        self.timeline.sliderReleased.connect(self.on_slider_released)
        timeline_layout.addWidget(self.timeline)
        
        # Frame info
        frame_info_layout = QHBoxLayout()
        self.frame_label = QLabel("Frame: 0/0")
        self.fps_label = QLabel("FPS: 0")
        frame_info_layout.addWidget(self.frame_label)
        frame_info_layout.addWidget(self.fps_label)
        frame_info_layout.addStretch()
        
        # Jump to frame
        frame_info_layout.addWidget(QLabel("Jump to:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.valueChanged.connect(self.jump_to_frame)
        frame_info_layout.addWidget(self.frame_spin)
        
        timeline_layout.addLayout(frame_info_layout)
        layout.addLayout(timeline_layout)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        # Play/Pause
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_btn)
        
        # Speed control
        controls_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x", "8x", "16x"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        controls_layout.addWidget(self.speed_combo)
        
        # Frame stepping
        self.prev_frame_btn = QPushButton("← Prev")
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        controls_layout.addWidget(self.prev_frame_btn)
        
        self.next_frame_btn = QPushButton("Next →")
        self.next_frame_btn.clicked.connect(self.next_frame)
        controls_layout.addWidget(self.next_frame_btn)
        
        controls_layout.addStretch()
        
        # Track ID assignment
        controls_layout.addWidget(QLabel("Assign Track ID:"))
        for i in range(1, 10):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda checked, tid=i: self.assign_track_id(tid))
            controls_layout.addWidget(btn)
            
        # Active tracks display
        self.active_tracks_label = QLabel("Active IDs: None")
        controls_layout.addWidget(self.active_tracks_label)
        
        layout.addLayout(controls_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # Frame navigation
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_frame)
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_playback)
        
        # Track ID assignment
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, lambda tid=i: self.assign_track_id(tid))
            
        # Save
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_changes)
        
    def setup_connections(self):
        """Connect signals"""
        self.video_decoder.frameReady.connect(self.on_frame_ready)
        self.video_decoder.positionChanged.connect(self.on_position_changed)
        self.video_canvas.bboxClicked.connect(self.on_bbox_clicked)
        
        # Start decoder thread
        self.video_decoder.start()
        
    def load_csv(self, csv_path: str):
        """Load video/bbox pairs from CSV"""
        df = pd.read_csv(csv_path)
        self.video_list = df.to_dict('records')
        
        if self.video_list:
            self.load_video_pair(0)
            
    def load_video_pair(self, idx: int):
        """Load a video and its corresponding bbox data"""
        if idx < 0 or idx >= len(self.video_list):
            return
            
        # Pause any playback before loading new video
        self.video_decoder.pause()
        self.play_btn.setText("Play")
        
        self.current_video_idx = idx
        pair = self.video_list[idx]
        
        video_path = pair['video_path']
        bbox_path = pair['bbox_json_path']
        
        print(f"\nLoading video {idx+1}/{len(self.video_list)}: {video_path}")
        print(f"Loading bbox data: {bbox_path}")
        
        # Load video with better error handling
        try:
            # Check if video file exists and is readable
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Check file size
            file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
            print(f"Video file size: {file_size:.1f} MB")
                
            # Try opening with cv2 first to check if it's valid
            test_cap = cv2.VideoCapture(video_path)
            if not test_cap.isOpened():
                test_cap.release()
                raise ValueError(f"Cannot open video file: {video_path}")
            test_cap.release()
            
            # Now load through our decoder
            fps, total_frames = self.video_decoder.load_video(video_path)
            self.timeline.setMaximum(max(0, total_frames - 1))
            self.frame_spin.setMaximum(max(0, total_frames - 1))
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Load bbox data
            self.bbox_manager.load_json(bbox_path)
            
            # Update UI
            self.file_label.setText(f"Video {idx+1}/{len(self.video_list)}: {os.path.basename(video_path)}")
            self.update_active_tracks()
            
            # Update navigation buttons
            self.prev_video_btn.setEnabled(idx > 0)
            self.next_video_btn.setEnabled(idx < len(self.video_list) - 1)
            
            # Mark timeline with detections
            self.mark_detections_on_timeline()
            
            # Seek to first frame to display it
            self.video_decoder.seek_to_frame(0)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Failed to load video/bbox:\n{str(e)}\n\nThis video may be corrupted or use an unsupported codec."
            
            # Show error and offer to skip
            reply = QMessageBox.question(
                self, "Error Loading Video",
                f"{error_msg}\n\nDo you want to skip to the next video?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes and idx < len(self.video_list) - 1:
                self.next_video()
            elif reply == QMessageBox.Yes and idx > 0:
                self.prev_video()
            
    def mark_detections_on_timeline(self):
        """Add visual markers for frames with detections"""
        frames_with_detections = self.bbox_manager.get_frames_with_detections()
        self.timeline.set_detection_frames(frames_with_detections)
        self.status_bar.showMessage(f"Frames with detections: {len(frames_with_detections)}")
        
    def on_frame_ready(self, frame: np.ndarray, frame_num: int):
        """Handle new frame from decoder"""
        # Display the frame
        self.video_canvas.display_frame(frame)
        
        # Get and display bboxes
        bboxes = self.bbox_manager.get_bboxes(frame_num)
        
        # Calculate scale factor if downsampling
        scale_x, scale_y = 1.0, 1.0
        if self.downsampling and hasattr(self.video_decoder, 'original_size') and self.video_decoder.original_size[0] > 0 and self.video_decoder.original_size[1] > 0:
            scale_x = self.video_decoder.target_size[0] / self.video_decoder.original_size[0]
            scale_y = self.video_decoder.target_size[1] / self.video_decoder.original_size[1]
            
        self.video_canvas.display_bboxes(bboxes, scale_x, scale_y)
        
    def on_position_changed(self, frame_num: int):
        """Update UI when frame position changes"""
        self.timeline.blockSignals(True)
        self.timeline.setValue(frame_num)
        self.timeline.blockSignals(False)
        
        total = self.video_decoder.total_frames
        self.frame_label.setText(f"Frame: {frame_num}/{total}")
        
    def on_bbox_clicked(self, bbox_idx: int):
        """Handle bbox selection"""
        frame_num = self.video_decoder.current_frame
        bboxes = self.bbox_manager.get_bboxes(frame_num)
        
        # Clear previous selection
        for bbox in bboxes:
            bbox.selected = False
            
        # Select clicked bbox
        if bbox_idx < len(bboxes):
            bboxes[bbox_idx].selected = True
            self.selected_bbox_idx = bbox_idx
            
        # Redraw
        if self.video_canvas.current_frame_array is not None:
            # Calculate scale factor if downsampling
            scale_x, scale_y = 1.0, 1.0
            if self.downsampling and hasattr(self.video_decoder, 'original_size') and self.video_decoder.original_size[0] > 0 and self.video_decoder.original_size[1] > 0:
                scale_x = self.video_decoder.target_size[0] / self.video_decoder.original_size[0]
                scale_y = self.video_decoder.target_size[1] / self.video_decoder.original_size[1]
            self.video_canvas.display_bboxes(bboxes, scale_x, scale_y)
        
    def assign_track_id(self, track_id: int):
        """Assign track ID to selected bbox"""
        if self.selected_bbox_idx is None:
            self.status_bar.showMessage("No bbox selected", 2000)
            return
            
        frame_num = self.video_decoder.current_frame
        self.bbox_manager.set_track_id(frame_num, self.selected_bbox_idx, track_id)
        
        # Force redraw
        if self.video_canvas.current_frame_array is not None:
            # Calculate scale factor if downsampling
            scale_x, scale_y = 1.0, 1.0
            if self.downsampling and hasattr(self.video_decoder, 'original_size') and self.video_decoder.original_size[0] > 0 and self.video_decoder.original_size[1] > 0:
                scale_x = self.video_decoder.target_size[0] / self.video_decoder.original_size[0]
                scale_y = self.video_decoder.target_size[1] / self.video_decoder.original_size[1]
            self.video_canvas.display_bboxes(bboxes, scale_x, scale_y)
        self.update_active_tracks()
        self.save_btn.setEnabled(True)
        
        self.status_bar.showMessage(f"Assigned track ID {track_id}", 2000)
        
    def update_active_tracks(self):
        """Update active tracks display"""
        active_ids = sorted(self.bbox_manager.active_track_ids)
        if active_ids:
            self.active_tracks_label.setText(f"Active IDs: {', '.join(map(str, active_ids))}")
        else:
            self.active_tracks_label.setText("Active IDs: None")
            
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.video_decoder.playing:
            self.video_decoder.pause()
            self.play_btn.setText("Play")
        else:
            self.video_decoder.play()
            self.play_btn.setText("Pause")
            
    def change_speed(self, speed_text: str):
        """Change playback speed"""
        speed = float(speed_text.rstrip('x'))
        self.video_decoder.set_speed(speed)
        
    def seek_to_frame(self, frame_num: int):
        """Seek to specific frame with scrubbing optimization"""
        if self.timeline.isSliderDown():
            # If dragging, use timer to throttle seeks
            self.pending_seek_frame = frame_num
            self.scrub_timer.stop()
            self.scrub_timer.start(50)  # 50ms delay for smooth scrubbing
        else:
            # Direct seek if not dragging
            self.video_decoder.seek_to_frame(frame_num)
            
    def perform_delayed_seek(self):
        """Perform the delayed seek after scrubbing"""
        if self.pending_seek_frame >= 0:
            self.video_decoder.seek_to_frame(self.pending_seek_frame)
            self.pending_seek_frame = -1
        
    def on_slider_released(self):
        """Handle slider release to ensure final seek"""
        self.scrub_timer.stop()
        if self.pending_seek_frame >= 0:
            self.video_decoder.seek_to_frame(self.pending_seek_frame)
            self.pending_seek_frame = -1
        else:
            # Ensure we're at the correct frame
            self.video_decoder.seek_to_frame(self.timeline.value())
            
    def jump_to_frame(self, frame_num: int):
        """Jump to frame from spinbox"""
        self.video_decoder.seek_to_frame(frame_num)
        
    def prev_frame(self):
        """Go to previous frame"""
        current = self.video_decoder.current_frame
        if current > 0:
            self.seek_to_frame(current - 1)
        
    def next_frame(self):
        """Go to next frame"""
        current = self.video_decoder.current_frame
        if current < self.video_decoder.total_frames - 1:
            self.seek_to_frame(current + 1)
            
    def prev_video(self):
        """Load previous video in list"""
        if self.current_video_idx > 0:
            if self.bbox_manager.modified:
                reply = QMessageBox.question(self, 'Unsaved Changes', 
                                           'Save changes before switching videos?',
                                           QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self.save_changes()
                elif reply == QMessageBox.Cancel:
                    return
            self.load_video_pair(self.current_video_idx - 1)
            
    def next_video(self):
        """Load next video in list"""
        if self.current_video_idx < len(self.video_list) - 1:
            if self.bbox_manager.modified:
                reply = QMessageBox.question(self, 'Unsaved Changes', 
                                           'Save changes before switching videos?',
                                           QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self.save_changes()
                elif reply == QMessageBox.Cancel:
                    return
            self.load_video_pair(self.current_video_idx + 1)
        
    def toggle_downsampling(self, checked: bool):
        """Toggle video downsampling"""
        self.downsampling = checked
        self.video_decoder.set_downsampling(checked)
        
        # Re-seek to current frame to apply
        self.video_decoder.seek_to_frame(self.video_decoder.current_frame)
        
    def save_changes(self):
        """Save bbox changes to JSON"""
        try:
            self.bbox_manager.save_json()
            self.save_btn.setEnabled(False)
            self.status_bar.showMessage("Changes saved successfully", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save: {str(e)}")
            
    def closeEvent(self, event):
        """Handle window close"""
        if self.bbox_manager.modified:
            reply = QMessageBox.question(
                self, 'Unsaved Changes',
                'Save changes before closing?',
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Yes:
                self.save_changes()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
                
        # Properly stop and cleanup video decoder
        self.video_decoder.stop_thread()
        self.video_decoder.quit()
        self.video_decoder.wait(2000)  # Wait up to 2 seconds
        if self.video_decoder.isRunning():
            self.video_decoder.terminate()  # Force terminate if still running
            
        event.accept()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Video Tracking Quality Control GUI")
    parser.add_argument('--csv', required=True, help='CSV file with video_path,bbox_json_path columns')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
        
    # Set OpenCV thread count to avoid FFmpeg threading issues
    cv2.setNumThreads(1)
        
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = TrackingQCWindow()
    window.load_csv(args.csv)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()