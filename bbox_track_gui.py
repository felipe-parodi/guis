#!/usr/bin/env python3
"""
Video Tracking Quality Control GUI - V3

A fast, streaming video player for reviewing and correcting object tracking predictions.
Handles long videos efficiently and provides interactive editing tools.

Usage:
    python bbox_track_gui.py --csv tracking_data.csv

CSV Format:
    video_path,bbox_json_path
    /path/to/video1.mp4,/path/to/bbox1.json

Requirements:
    pip install PyQt5 opencv-python-headless numpy pandas

Keybindings:
    - Space: Play/Pause
    - Left/Right Arrow: Step one frame
    - 1-9: Assign track ID to selected box
    - A: Enter 'Add Bbox' mode. Click and drag to draw a new box.
    - Delete: Delete the selected bbox.
    - Ctrl+S: Save changes.
    - I: Interpolate selected track
    - Q/E: Jump to prev/next detection
    - T: Start tracking selected box forward
"""

import sys
import os

# Set environment variable before importing cv2 to avoid threading issues
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "8192"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"

import argparse
import threading
import time
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Import our utility modules
from utils import BBox, BBoxManager, ShortTermTracker, VisualizationManager
from utils.export_utils import export_tracking_data


class VideoDecoder(QThread):
    """Efficient video decoder thread with frame caching."""
    frameReady = pyqtSignal(object, int)  # Use object to handle potential None frames
    positionChanged = pyqtSignal(int)
    videoLoaded = pyqtSignal(float, int)

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
        self.target_size = (960, 540)
        self.force_refresh = False
        self.is_loading = False
        self.should_stop = False
        
        # Frame caching for double-buffering
        self.frame_cache = {}
        self.cache_size = 10  # Cache last 10 frames
        
    def get_video_info(self) -> Dict:
        """Get video metadata for export purposes."""
        if not self.cap:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.fps,
            'total_frames': self.total_frames,
            'name': os.path.basename(self.video_path) if self.video_path else 'video'
        }

    def load_video(self, video_path: str):
        self.is_loading = True
        self.playing = False
        self.msleep(100)

        with self.seek_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

            self.video_path = video_path
            self.frame_cache.clear()  # Clear cache when loading new video
            
            try:
                self.cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    raise ValueError(f"Cannot open video: {video_path}")

                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.current_frame = 0

                if self.total_frames <= 0:
                    print(f"Warning: Could not get frame count for {video_path}. "
                          f"This might be a live stream or corrupted file.")
                    self.total_frames = 1_000_000  # Large default

                ret, frame = self.cap.read()
                if ret:
                    self.original_size = (frame.shape[1], frame.shape[0])
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.videoLoaded.emit(self.fps, self.total_frames)
                    print(f"Video loaded: {self.original_size[0]}x{self.original_size[1]}, "
                          f"{self.total_frames} frames @ {self.fps:.2f} fps")
                else:
                    raise ValueError(f"Cannot read first frame from video: {video_path}")
            except Exception as e:
                self.is_loading = False
                self.frameReady.emit(None, -1)  # Signal failure
                raise

        self.is_loading = False
        self.seek_to_frame(0)

    def seek_to_frame(self, frame_num: int):
        if self.is_loading:
            return
        with self.seek_lock:
            if self.cap and self.cap.isOpened():
                self.seek_frame = max(0, min(frame_num, self.total_frames - 1))
                self.force_refresh = True

    def set_downsampling(self, enabled: bool):
        self.downsampling = enabled
        self.force_refresh = True
        
    def get_cached_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """Get frame from cache if available."""
        return self.frame_cache.get(frame_num)
    
    def cache_frame(self, frame_num: int, frame: np.ndarray):
        """Cache a frame, removing old frames if cache is full."""
        self.frame_cache[frame_num] = frame.copy()
        
        # Remove old frames if cache is full
        if len(self.frame_cache) > self.cache_size:
            oldest_frame = min(self.frame_cache.keys())
            del self.frame_cache[oldest_frame]

    def run(self):
        last_frame_time = time.time()
        while not self.should_stop:
            if self.is_loading or not self.cap or not self.cap.isOpened():
                self.msleep(50)
                continue

            frame_time = 1.0 / (self.fps * self.playback_speed) if self.fps > 0 else 1/30

            seeked = False
            with self.seek_lock:
                if self.seek_frame != -1:
                    # Check cache first
                    cached_frame = self.get_cached_frame(self.seek_frame)
                    if cached_frame is not None and not self.downsampling:
                        # Use cached frame
                        self.current_frame = self.seek_frame
                        self.frameReady.emit(cached_frame, self.current_frame)
                        self.positionChanged.emit(self.current_frame)
                        self.seek_frame = -1
                        last_frame_time = time.time()
                        continue
                    
                    # Not in cache, seek normally
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.seek_frame)
                    self.current_frame = self.seek_frame
                    self.seek_frame = -1
                    last_frame_time = time.time()
                    seeked = True

            if (self.playing or seeked or self.force_refresh):
                if self.playing and not seeked and not self.force_refresh:
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_time:
                        self.msleep(int((frame_time - elapsed) * 1000))
                        continue

                ret, frame = self.cap.read()
                if ret:
                    # Cache the frame
                    self.cache_frame(self.current_frame, frame)
                    
                    if self.downsampling:
                        frame = cv2.resize(frame, self.target_size, 
                                         interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frameReady.emit(frame, self.current_frame)
                    if self.playing:
                        self.current_frame += 1
                    self.positionChanged.emit(self.current_frame)
                    last_frame_time = time.time()
                    self.force_refresh = False
                else:
                    if self.playing:
                        self.playing = False
                        print("End of video reached.")
            else:
                self.msleep(20)

    def stop_thread(self):
        self.should_stop = True
        self.playing = False
        with self.seek_lock:
            if self.cap:
                self.cap.release()
                self.cap = None

    def play(self): 
        self.playing = True
        
    def pause(self): 
        self.playing = False
        
    def set_speed(self, speed: float): 
        self.playback_speed = speed


class DetectionTimeline(QSlider):
    """Enhanced timeline with detection markers."""
    def __init__(self, orientation=Qt.Horizontal):
        super().__init__(orientation)
        self.detection_frames = []
        self.setStyleSheet("""
            QSlider::groove:horizontal { 
                border: 1px solid #bbb; background: #ddd; height: 10px; border-radius: 4px; 
            }
            QSlider::sub-page:horizontal { 
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #66e, stop: 1 #bbf); 
                border: 1px solid #777; height: 10px; border-radius: 4px; 
            }
            QSlider::handle:horizontal { 
                background: #fff; border: 1px solid #777; width: 14px; height: 14px; 
                margin: -4px 0; border-radius: 7px; 
            }
        """)
    
    def set_detection_frames(self, frames: List[int]):
        self.detection_frames = frames
        self.update()
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.detection_frames or self.maximum() == 0:
            return
        
        painter = QPainter(self)
        groove_rect = self.style().subControlRect(
            QStyle.CC_Slider, QStyleOptionSlider(), QStyle.SC_SliderGroove, self
        )
        
        pen = QPen(QColor(255, 0, 0, 150), 1)
        painter.setPen(pen)
        
        for frame in self.detection_frames:
            pos_ratio = frame / self.maximum()
            x = groove_rect.left() + pos_ratio * groove_rect.width()
            painter.drawLine(int(x), groove_rect.top(), int(x), groove_rect.bottom())


class InteractiveBBoxItem(QGraphicsRectItem):
    """Enhanced bbox item with better visual feedback."""
    bboxChanged = pyqtSignal(int, QRectF)
    bboxSelected = pyqtSignal(int)
    
    HandleSize = 8
    
    def __init__(self, rect: QRectF, bbox_obj: BBox):
        super().__init__(rect)
        self.bbox_obj = bbox_obj
        self.handles = {}
        self.handle_positions = {
            0: "TopLeft", 
            1: "Top", 
            2: "TopRight",
            3: "Left", 
            4: "Right",
            5: "BottomLeft", 
            6: "Bottom", 
            7: "BottomRight",
        }
        self.selected_handle = None
        self.mouse_press_pos = None
        self.mouse_press_rect = None

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)

        self.update_appearance()

    def update_appearance(self, is_suspicious=False):
        pen = QPen()
        pen.setWidth(2)
        if self.isSelected():
            pen.setColor(Qt.yellow)
            pen.setStyle(Qt.DashLine)
        elif is_suspicious:
            pen.setColor(Qt.red)
            pen.setStyle(Qt.DashDotLine)
            pen.setWidth(3)
        elif self.bbox_obj.track_id is not None:
            colors = [Qt.cyan, Qt.magenta, Qt.green, Qt.red, Qt.blue, QColor("#FFA500")]
            pen.setColor(colors[self.bbox_obj.track_id % len(colors)])
        else:
            pen.setColor(Qt.white)
        self.setPen(pen)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent):
        handle = self.get_handle_at(event.pos())
        if handle is not None:
            cursor = (Qt.SizeFDiagCursor if handle in (0, 2, 5, 7) else 
                     (Qt.SizeVerCursor if handle in (1, 6) else Qt.SizeHorCursor))
            self.setCursor(cursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        self.selected_handle = self.get_handle_at(event.pos())
        self.mouse_press_pos = event.pos()
        self.mouse_press_rect = self.rect()
        self.bboxSelected.emit(self.bbox_obj.instance_id)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        if self.selected_handle is not None:
            self.interactive_resize(event.pos())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        self.selected_handle = None
        super().mouseReleaseEvent(event)
        # Finalize change
        self.bboxChanged.emit(self.bbox_obj.instance_id, self.rect())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            # Called after a move
            self.bboxChanged.emit(self.bbox_obj.instance_id, 
                                self.rect().translated(value))
        return super().itemChange(change, value)

    def get_handle_rects(self):
        s = self.HandleSize
        r = self.rect()
        return [
            QRectF(r.left(), r.top(), s, s), 
            QRectF(r.center().x() - s/2, r.top(), s, s), 
            QRectF(r.right() - s, r.top(), s, s),
            QRectF(r.left(), r.center().y() - s/2, s, s), 
            QRectF(r.right() - s, r.center().y() - s/2, s, s),
            QRectF(r.left(), r.bottom() - s, s, s), 
            QRectF(r.center().x() - s/2, r.bottom() - s, s, s), 
            QRectF(r.right() - s, r.bottom() - s, s, s),
        ]

    def get_handle_at(self, pos):
        for i, r in enumerate(self.get_handle_rects()):
            if r.contains(pos):
                return i
        return None

    def interactive_resize(self, pos):
        rect = self.rect()
        diff = pos - self.mouse_press_pos
        
        if self.selected_handle == 0: 
            rect.setTopLeft(self.mouse_press_rect.topLeft() + diff)
        elif self.selected_handle == 1: 
            rect.setTop(self.mouse_press_rect.top() + diff.y())
        elif self.selected_handle == 2: 
            rect.setTopRight(self.mouse_press_rect.topRight() + diff)
        elif self.selected_handle == 3: 
            rect.setLeft(self.mouse_press_rect.left() + diff.x())
        elif self.selected_handle == 4: 
            rect.setRight(self.mouse_press_rect.right() + diff.x())
        elif self.selected_handle == 5: 
            rect.setBottomLeft(self.mouse_press_rect.bottomLeft() + diff)
        elif self.selected_handle == 6: 
            rect.setBottom(self.mouse_press_rect.bottom() + diff.y())
        elif self.selected_handle == 7: 
            rect.setBottomRight(self.mouse_press_rect.bottomRight() + diff)
        
        self.setRect(rect.normalized())

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        if self.isSelected():
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(Qt.yellow)
            painter.setPen(QPen(Qt.black, 1))
            for r in self.get_handle_rects():
                painter.drawRect(r)


class VideoCanvas(QGraphicsView):
    """Enhanced video canvas with visualization options."""
    bboxAdded = pyqtSignal(QRectF)
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.is_adding_bbox = False
        self.visualization_items = []  # Track visualization overlays

    def display_frame(self, frame: np.ndarray):
        height, width = frame.shape[:2]
        q_image = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        if self.scene.sceneRect().isEmpty():
            self.setSceneRect(QRectF(pixmap.rect()))
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def display_bboxes(self, bboxes: List[BBox], scale: QPointF, 
                       selected_instance_id: Optional[int], 
                       confidence_threshold: float = 0.0, 
                       draw_labels: bool = True, 
                       track_trails: List[Tuple[BBox, float]] = None,
                       visualization_items: List = None):
        # Clear only bboxes and old visualizations, not the pixmap
        for item in self.scene.items():
            if (isinstance(item, InteractiveBBoxItem) or 
                isinstance(item, QGraphicsTextItem) or
                item in self.visualization_items):
                self.scene.removeItem(item)
        
        self.visualization_items.clear()
        
        # Add new visualization items
        if visualization_items:
            for item in visualization_items:
                self.scene.addItem(item)
                self.visualization_items.append(item)
        
        # Draw track trails first (behind current boxes)
        if track_trails:
            for trail_bbox, opacity in track_trails:
                rect = QRectF(trail_bbox.x1 * scale.x(), trail_bbox.y1 * scale.y(),
                              (trail_bbox.x2 - trail_bbox.x1) * scale.x(), 
                              (trail_bbox.y2 - trail_bbox.y1) * scale.y())
                trail_item = QGraphicsRectItem(rect)
                pen = QPen(Qt.red, 1, Qt.DotLine)
                trail_item.setPen(pen)
                trail_item.setOpacity(opacity)
                self.scene.addItem(trail_item)
                self.visualization_items.append(trail_item)
        
        for bbox in bboxes:
            # Apply confidence filtering with opacity
            opacity = 1.0
            if bbox.confidence is not None and bbox.confidence < confidence_threshold:
                opacity = 0.3  # Reduced opacity for filtered boxes
            
            rect = QRectF(bbox.x1 * scale.x(), bbox.y1 * scale.y(),
                          (bbox.x2 - bbox.x1) * scale.x(), 
                          (bbox.y2 - bbox.y1) * scale.y())
            
            item = InteractiveBBoxItem(rect, bbox)
            item.setSelected(bbox.instance_id == selected_instance_id)
            item.bboxChanged.connect(self.parent().on_bbox_changed)
            item.bboxSelected.connect(self.parent().on_bbox_selected)
            item.setOpacity(opacity)
            
            # Check if track is suspicious (for validation highlighting)
            is_suspicious = (hasattr(self.parent(), 'is_track_suspicious') and
                           self.parent().is_track_suspicious(bbox))
            item.update_appearance(is_suspicious)
            
            self.scene.addItem(item)
            
            if draw_labels:
                label_text = f"ID:{bbox.track_id}" if bbox.track_id is not None else "ID:?"
                if bbox.confidence: 
                    label_text += f" ({bbox.confidence:.2f})"
                
                text_item = QGraphicsTextItem(label_text, item)
                text_item.setDefaultTextColor(item.pen().color())
                text_item.setPos(rect.left(), 
                               rect.top() - text_item.boundingRect().height())
                text_item.setOpacity(opacity)

    def set_add_bbox_mode(self, enabled: bool):
        self.is_adding_bbox = enabled
        self.setDragMode(QGraphicsView.RubberBandDrag if enabled else QGraphicsView.NoDrag)
        self.viewport().setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        super().mouseReleaseEvent(event)
        if self.is_adding_bbox and event.button() == Qt.LeftButton:
            selection_rect = self.scene.selectionArea().boundingRect()
            if not selection_rect.isEmpty():
                self.bboxAdded.emit(selection_rect)
            self.set_add_bbox_mode(False)  # Exit mode after drawing


class TrackingQCWindow(QMainWindow):
    """Enhanced main window with tracking and visualization features."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Tracking QC Tool - Enhanced")
        self.setGeometry(100, 100, 1800, 1000)
        
        self.video_decoder = VideoDecoder()
        self.bbox_manager = BBoxManager()
        self.viz_manager = VisualizationManager(self.bbox_manager)
        self.tracker = ShortTermTracker()
        
        self.current_video_idx = 0
        self.video_list = []
        self.selected_bbox_instance_id = None
        self.downsampling = False
        self.confidence_threshold = 0.0
        self.show_track_trails = False
        self.show_trajectories = False
        self.show_heatmap = False
        self.auto_save_enabled = False
        self.label_draw_threshold = 15  # FPS threshold for label drawing
        self.tracking_active = False  # Track if we're currently tracking
        
        self.scrub_timer = QTimer(self)
        self.scrub_timer.setSingleShot(True)
        self.scrub_timer.timeout.connect(self.perform_delayed_seek)
        self.pending_seek_frame = -1
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Enhanced toolbar
        toolbar = QHBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.prev_video_btn = QPushButton("← Prev")
        self.next_video_btn = QPushButton("Next →")
        self.downsample_cb = QCheckBox("Downsample")
        self.add_bbox_btn = QPushButton("Add Bbox (A)")
        self.add_bbox_btn.setCheckable(True)
        self.save_btn = QPushButton("Save (Ctrl+S)")
        self.save_btn.setStyleSheet("background-color: #c8e6c9;")
        
        # Export options
        self.export_btn = QPushButton("Export...")
        self.export_menu = QMenu()
        self.export_menu.addAction("Export to COCO", lambda: self.export_data('coco'))
        self.export_menu.addAction("Export to YOLO", lambda: self.export_data('yolo'))
        self.export_menu.addAction("Export to CSV", lambda: self.export_data('csv'))
        self.export_btn.setMenu(self.export_menu)
        
        toolbar.addWidget(self.file_label)
        toolbar.addStretch()
        toolbar.addWidget(self.prev_video_btn)
        toolbar.addWidget(self.next_video_btn)
        toolbar.addWidget(self.downsample_cb)
        toolbar.addWidget(self.add_bbox_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.export_btn)
        main_layout.addLayout(toolbar)

        # Main content with enhanced sidebar
        content_layout = QHBoxLayout()
        self.video_canvas = VideoCanvas()
        content_layout.addWidget(self.video_canvas, 3)

        # Enhanced right sidebar
        sidebar = QVBoxLayout()
        
        # Playback controls
        sidebar.addWidget(QLabel("Playback Controls"))
        self.play_btn = QPushButton("Play (Space)")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.speed_combo.setCurrentIndex(2)
        
        playback_layout = QHBoxLayout()
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(QLabel("Speed:"))
        playback_layout.addWidget(self.speed_combo)
        sidebar.addLayout(playback_layout)

        # Frame stepping
        step_layout = QHBoxLayout()
        self.prev_frame_btn = QPushButton("← Prev Frame")
        self.next_frame_btn = QPushButton("Next Frame →")
        step_layout.addWidget(self.prev_frame_btn)
        step_layout.addWidget(self.next_frame_btn)
        sidebar.addLayout(step_layout)

        # Track ID assignment
        id_layout = QGridLayout()
        id_layout.addWidget(QLabel("Assign Track ID:"), 0, 0, 1, 3)
        for i in range(1, 10):
            btn = QPushButton(str(i))
            btn.clicked.connect(lambda _, tid=i: self.assign_track_id(tid))
            id_layout.addWidget(btn, (i-1)//3 + 1, (i-1)%3)
        sidebar.addLayout(id_layout)

        # Enhanced tracking controls
        sidebar.addWidget(QLabel("Tracking Controls"))
        self.interpolate_btn = QPushButton("Interpolate Track (I)")
        self.track_forward_btn = QPushButton("Track Forward (T)")
        self.stop_tracking_btn = QPushButton("Stop Tracking")
        self.stop_tracking_btn.setEnabled(False)
        
        track_layout = QVBoxLayout()
        track_layout.addWidget(self.interpolate_btn)
        track_layout.addWidget(self.track_forward_btn)
        track_layout.addWidget(self.stop_tracking_btn)
        sidebar.addLayout(track_layout)
        
        # Tracker type selection
        tracker_layout = QHBoxLayout()
        tracker_layout.addWidget(QLabel("Tracker:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["csrt", "kcf", "mosse", "mil"])
        tracker_layout.addWidget(self.tracker_combo)
        sidebar.addLayout(tracker_layout)
        
        # Confidence threshold slider
        sidebar.addWidget(QLabel("Confidence Filter:"))
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(0)
        self.conf_slider.valueChanged.connect(self.update_confidence_threshold)
        self.conf_label = QLabel("0.00")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        sidebar.addLayout(conf_layout)
        
        # Visualization options
        sidebar.addWidget(QLabel("Visualization:"))
        self.track_trails_cb = QCheckBox("Show Track Trails")
        self.track_trails_cb.toggled.connect(self.toggle_track_trails)
        self.trajectories_cb = QCheckBox("Show Trajectories")
        self.trajectories_cb.toggled.connect(self.toggle_trajectories)
        self.heatmap_cb = QCheckBox("Show Heatmap")
        self.heatmap_cb.toggled.connect(self.toggle_heatmap)
        
        sidebar.addWidget(self.track_trails_cb)
        sidebar.addWidget(self.trajectories_cb)
        sidebar.addWidget(self.heatmap_cb)
        
        # Auto-save and performance options
        sidebar.addWidget(QLabel("Options:"))
        self.auto_save_cb = QCheckBox("Auto-save")
        self.auto_save_cb.toggled.connect(self.toggle_auto_save)
        sidebar.addWidget(self.auto_save_cb)
        
        # Label draw threshold
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("Label FPS limit:"))
        self.label_fps_spin = QSpinBox()
        self.label_fps_spin.setRange(0, 60)
        self.label_fps_spin.setValue(15)
        self.label_fps_spin.valueChanged.connect(self.update_label_threshold)
        label_layout.addWidget(self.label_fps_spin)
        sidebar.addLayout(label_layout)

        self.active_tracks_label = QLabel("Active IDs:")
        self.active_tracks_label.setWordWrap(True)
        sidebar.addWidget(self.active_tracks_label)
        sidebar.addStretch()
        
        # Auto-save timer
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.auto_save)
        self.autosave_timer.setSingleShot(True)
        
        content_layout.addLayout(sidebar, 1)
        main_layout.addLayout(content_layout)

        # Timeline
        timeline_layout = QVBoxLayout()
        self.timeline = DetectionTimeline()
        self.frame_label = QLabel("Frame: 0 / 0")
        self.fps_label = QLabel("FPS: 0.0")
        frame_info_layout = QHBoxLayout()
        frame_info_layout.addWidget(self.frame_label)
        frame_info_layout.addStretch()
        frame_info_layout.addWidget(self.fps_label)
        timeline_layout.addLayout(frame_info_layout)
        timeline_layout.addWidget(self.timeline)
        main_layout.addLayout(timeline_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.setup_shortcuts()

    def setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_Left), self, self.prev_frame)
        QShortcut(QKeySequence(Qt.Key_Right), self, self.next_frame)
        QShortcut(QKeySequence(Qt.Key_Delete), self, self.delete_selected_bbox)
        QShortcut(QKeySequence("A"), self, self.add_bbox_btn.toggle)
        QShortcut(QKeySequence("Ctrl+S"), self, self.save_changes)
        QShortcut(QKeySequence("I"), self, self.interpolate_selected_track)
        QShortcut(QKeySequence("Q"), self, self.jump_to_prev_detection)
        QShortcut(QKeySequence("E"), self, self.jump_to_next_detection)
        QShortcut(QKeySequence("T"), self, self.start_tracking_forward)
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, 
                     lambda tid=i: self.assign_track_id(tid))

    def setup_connections(self):
        self.video_decoder.frameReady.connect(self.on_frame_ready)
        self.video_decoder.videoLoaded.connect(self.on_video_loaded)
        self.video_decoder.positionChanged.connect(self.on_position_changed)
        self.video_decoder.start()

        self.prev_video_btn.clicked.connect(self.prev_video)
        self.next_video_btn.clicked.connect(self.next_video)
        self.save_btn.clicked.connect(self.save_changes)
        self.downsample_cb.toggled.connect(self.toggle_downsampling)
        
        self.play_btn.clicked.connect(self.toggle_playback)
        self.speed_combo.currentTextChanged.connect(self.change_speed)
        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        
        self.timeline.valueChanged.connect(self.seek_from_slider)
        self.timeline.sliderReleased.connect(self.on_slider_released)

        self.add_bbox_btn.toggled.connect(self.video_canvas.set_add_bbox_mode)
        self.video_canvas.bboxAdded.connect(self.on_bbox_added)
        self.interpolate_btn.clicked.connect(self.interpolate_selected_track)
        
        # New tracking connections
        self.track_forward_btn.clicked.connect(self.start_tracking_forward)
        self.stop_tracking_btn.clicked.connect(self.stop_tracking)
        self.tracker_combo.currentTextChanged.connect(self.change_tracker_type)

    def export_data(self, format_type: str):
        """Export tracking data to various formats."""
        if not self.bbox_manager.bbox_data:
            QMessageBox.warning(self, "Export Warning", "No tracking data to export.")
            return
        
        # Get save location
        if format_type == 'coco':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export COCO Annotations", "", "JSON Files (*.json)"
            )
        elif format_type == 'yolo':
            file_path = QFileDialog.getExistingDirectory(
                self, "Export YOLO Annotations Directory"
            )
        elif format_type == 'csv':
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export CSV Annotations", "", "CSV Files (*.csv)"
            )
        
        if not file_path:
            return
        
        # Get video info for export
        video_info = self.video_decoder.get_video_info()
        
        try:
            success = export_tracking_data(
                self.bbox_manager, 
                file_path, 
                format_type, 
                video_info
            )
            
            if success:
                QMessageBox.information(
                    self, "Export Successful", 
                    f"Data exported to {format_type.upper()} format successfully!"
                )
            else:
                QMessageBox.critical(
                    self, "Export Failed", 
                    f"Failed to export data to {format_type.upper()} format."
                )
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", 
                f"Error during export: {str(e)}"
            )

    def start_tracking_forward(self):
        """Start tracking selected bbox forward."""
        if not self.selected_bbox_instance_id:
            self.status_bar.showMessage("Select a bbox first", 2000)
            return
        
        current_frame = self.video_decoder.current_frame
        bbox = next((b for b in self.bbox_manager.get_bboxes(current_frame) 
                     if b.instance_id == self.selected_bbox_instance_id), None)
        
        if not bbox:
            self.status_bar.showMessage("Selected bbox not found", 2000)
            return
        
        # Get current frame
        cached_frame = self.video_decoder.get_cached_frame(current_frame)
        if cached_frame is None:
            self.status_bar.showMessage("Frame not available for tracking", 2000)
            return
        
        # Convert RGB to BGR for OpenCV
        cv_frame = cv2.cvtColor(cached_frame, cv2.COLOR_RGB2BGR)
        
        # Initialize tracker
        tracker_type = self.tracker_combo.currentText()
        self.tracker = ShortTermTracker(tracker_type)
        
        if self.tracker.init(cv_frame, bbox):
            self.tracking_active = True
            self.track_forward_btn.setEnabled(False)
            self.stop_tracking_btn.setEnabled(True)
            self.status_bar.showMessage(f"Started {tracker_type.upper()} tracking", 3000)
        else:
            self.status_bar.showMessage("Failed to initialize tracker", 2000)

    def stop_tracking(self):
        """Stop active tracking."""
        self.tracking_active = False
        self.tracker.reset()
        self.track_forward_btn.setEnabled(True)
        self.stop_tracking_btn.setEnabled(False)
        self.status_bar.showMessage("Tracking stopped", 2000)

    def change_tracker_type(self, tracker_type: str):
        """Change the tracker type."""
        if self.tracking_active:
            self.stop_tracking()

    def update_tracked_bbox(self):
        """Update bbox using tracker if active."""
        if not self.tracking_active or not self.selected_bbox_instance_id:
            return
        
        current_frame = self.video_decoder.current_frame
        cached_frame = self.video_decoder.get_cached_frame(current_frame)
        
        if cached_frame is None:
            return
        
        # Convert RGB to BGR for OpenCV
        cv_frame = cv2.cvtColor(cached_frame, cv2.COLOR_RGB2BGR)
        
        # Update tracker
        tracked_bbox = self.tracker.update(cv_frame)
        
        if tracked_bbox:
            # Update the bbox in manager
            tracked_bbox.track_id = self.get_selected_track_id()
            tracked_bbox.instance_id = self.selected_bbox_instance_id
            
            self.bbox_manager.update_bbox(
                current_frame, 
                self.selected_bbox_instance_id,
                tracked_bbox.x1, tracked_bbox.y1, 
                tracked_bbox.x2, tracked_bbox.y2
            )
            
            self.force_redraw_current_frame()
        else:
            # Tracking failed
            self.stop_tracking()
            self.status_bar.showMessage("Tracking lost", 2000)

    def get_selected_track_id(self) -> Optional[int]:
        """Get track ID of currently selected bbox."""
        if not self.selected_bbox_instance_id:
            return None
        
        current_frame = self.video_decoder.current_frame
        bbox = next((b for b in self.bbox_manager.get_bboxes(current_frame) 
                     if b.instance_id == self.selected_bbox_instance_id), None)
        
        return bbox.track_id if bbox else None

    def toggle_trajectories(self, enabled):
        """Toggle trajectory visualization."""
        self.show_trajectories = enabled
        self.viz_manager.show_trajectories = enabled
        self.force_redraw_current_frame()

    def toggle_heatmap(self, enabled):
        """Toggle heatmap visualization."""
        self.show_heatmap = enabled
        self.viz_manager.show_heatmap = enabled
        self.force_redraw_current_frame()

    def load_csv(self, csv_path: str):
        """Load video list from CSV file."""
        import pandas as pd
        self.video_list = pd.read_csv(csv_path).to_dict('records')
        if self.video_list:
            self.load_video_pair(0)

    def load_video_pair(self, idx: int):
        """Load video and corresponding annotation file."""
        if not (0 <= idx < len(self.video_list)):
            return
        self.check_unsaved_changes()

        self.current_video_idx = idx
        pair = self.video_list[idx]
        video_path, bbox_path = pair['video_path'], pair['bbox_json_path']
        
        self.status_bar.showMessage(f"Loading video: {os.path.basename(video_path)}...")
        self.file_label.setText(f"{idx+1}/{len(self.video_list)}: {os.path.basename(video_path)}")
        self.selected_bbox_instance_id = None
        
        # Stop any active tracking
        if self.tracking_active:
            self.stop_tracking()
        
        try:
            self.bbox_manager.load_json(bbox_path)
            self.video_decoder.load_video(video_path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", 
                               f"Failed to load data for {video_path}.\n{e}")

    def on_video_loaded(self, fps: float, total_frames: int):
        """Handle video loaded signal."""
        self.timeline.setMaximum(total_frames - 1)
        self.fps_label.setText(f"FPS: {fps:.2f}")
        self.update_active_tracks()
        self.timeline.set_detection_frames(self.bbox_manager.get_frames_with_detections())
        self.update_ui_state()

    def update_ui_state(self):
        """Update UI elements based on current state."""
        self.prev_video_btn.setEnabled(self.current_video_idx > 0)
        self.next_video_btn.setEnabled(self.current_video_idx < len(self.video_list) - 1)
        self.save_btn.setEnabled(self.bbox_manager.modified)
        
        if self.bbox_manager.modified:
            self.save_btn.setText("Save* (Ctrl+S)")
            self.save_btn.setStyleSheet("background-color: #ffcdd2;")  # Reddish for unsaved
            # Restart auto-save timer if enabled
            if self.auto_save_enabled:
                self.autosave_timer.stop()
                self.autosave_timer.start(30000)  # 30 seconds
        else:
            self.save_btn.setText("Save (Ctrl+S)")
            self.save_btn.setStyleSheet("background-color: #c8e6c9;")  # Greenish for saved

    def on_frame_ready(self, frame: Optional[np.ndarray], frame_num: int):
        """Handle new frame from video decoder."""
        if frame is None:
            QMessageBox.critical(self, "Playback Error", 
                               "Failed to decode frame. Video may be corrupt.")
            self.video_decoder.pause()
            return

        # Update tracker if active
        if self.tracking_active:
            self.update_tracked_bbox()

        self.video_canvas.display_frame(frame)
        bboxes = self.bbox_manager.get_bboxes(frame_num)
        
        scale_x, scale_y = 1.0, 1.0
        if self.downsampling and hasattr(self.video_decoder, 'original_size'):
            orig_w, orig_h = self.video_decoder.original_size
            target_w, target_h = self.video_decoder.target_size
            if orig_w > 0 and orig_h > 0:
                scale_x, scale_y = target_w / orig_w, target_h / orig_h
        
        # Check if we should draw labels based on FPS threshold
        is_playing = self.video_decoder.playing
        current_fps = self.fps_label.text().split(': ')[1].split()[0] if ': ' in self.fps_label.text() else "0"
        try:
            fps_val = float(current_fps)
        except:
            fps_val = 0
        draw_labels = not is_playing or fps_val < self.label_draw_threshold or len(bboxes) < 10
        
        # Get track trails if enabled
        track_trails = None
        if self.show_track_trails and self.selected_bbox_instance_id:
            selected_bbox = next((b for b in bboxes if b.instance_id == self.selected_bbox_instance_id), None)
            if selected_bbox and selected_bbox.track_id is not None:
                track_trails = self.get_track_trail(selected_bbox.track_id, frame_num)
        
        # Get visualization items
        visualization_items = []
        if self.show_trajectories and self.selected_bbox_instance_id:
            selected_bbox = next((b for b in bboxes if b.instance_id == self.selected_bbox_instance_id), None)
            if selected_bbox and selected_bbox.track_id is not None:
                traj_path = self.viz_manager.get_track_trajectory_path(
                    selected_bbox.track_id, frame_num, QPointF(scale_x, scale_y)
                )
                if traj_path:
                    visualization_items.append(traj_path)
                    
                traj_points = self.viz_manager.get_trajectory_points(
                    selected_bbox.track_id, frame_num, QPointF(scale_x, scale_y)
                )
                visualization_items.extend(traj_points)
        
        self.video_canvas.display_bboxes(
            bboxes, QPointF(scale_x, scale_y), 
            self.selected_bbox_instance_id,
            self.confidence_threshold,
            draw_labels,
            track_trails,
            visualization_items
        )

    def on_position_changed(self, frame_num: int):
        """Handle frame position change."""
        self.timeline.blockSignals(True)
        self.timeline.setValue(frame_num)
        self.timeline.blockSignals(False)
        self.frame_label.setText(f"Frame: {frame_num} / {self.video_decoder.total_frames - 1}")
    
    def on_bbox_selected(self, instance_id: int):
        """Handle bbox selection."""
        self.selected_bbox_instance_id = instance_id
        self.force_redraw_current_frame()

    def on_bbox_changed(self, instance_id: int, new_rect: QRectF):
        """Handle bbox coordinate changes."""
        scale_x, scale_y = 1.0, 1.0
        if self.downsampling and hasattr(self.video_decoder, 'original_size'):
            orig_w, orig_h = self.video_decoder.original_size
            target_w, target_h = self.video_decoder.target_size
            if orig_w > 0 and orig_h > 0:
                scale_x, scale_y = orig_w / target_w, orig_h / target_h
        
        # Convert coordinates back to original scale
        self.bbox_manager.update_bbox(
            self.video_decoder.current_frame, 
            instance_id,
            new_rect.x() * scale_x, new_rect.y() * scale_y,
            (new_rect.x() + new_rect.width()) * scale_x, 
            (new_rect.y() + new_rect.height()) * scale_y
        )
        self.update_ui_state()
        
    def on_bbox_added(self, rect: QRectF):
        """Handle new bbox addition."""
        scale_x, scale_y = 1.0, 1.0
        if self.downsampling and hasattr(self.video_decoder, 'original_size'):
            orig_w, orig_h = self.video_decoder.original_size
            target_w, target_h = self.video_decoder.target_size
            if orig_w > 0 and orig_h > 0:
                scale_x, scale_y = orig_w / target_w, orig_h / target_h
        
        # Convert to original coordinates
        x1 = rect.x() * scale_x
        y1 = rect.y() * scale_y
        x2 = (rect.x() + rect.width()) * scale_x
        y2 = (rect.y() + rect.height()) * scale_y

        # Prompt for track ID
        tid, ok = QInputDialog.getInt(self, "Assign Track ID", 
                                     "Enter Track ID for new box:", 1, 0, 999, 1)
        if ok:
            new_bbox = BBox(x1, y1, x2, y2, track_id=tid, confidence=1.0)
            self.bbox_manager.add_bbox(self.video_decoder.current_frame, new_bbox)
            self.selected_bbox_instance_id = new_bbox.instance_id
            self.update_active_tracks()
            self.force_redraw_current_frame()
            self.update_ui_state()

    def assign_track_id(self, track_id: int):
        """Assign track ID to selected bbox."""
        if self.selected_bbox_instance_id is None:
            self.status_bar.showMessage("No bbox selected.", 2000)
            return
        
        current_frame = self.video_decoder.current_frame
        bbox_to_update = next(
            (b for b in self.bbox_manager.get_bboxes(current_frame) 
             if b.instance_id == self.selected_bbox_instance_id), None
        )
        if bbox_to_update:
            self.bbox_manager.update_bbox(
                current_frame, 
                self.selected_bbox_instance_id,
                bbox_to_update.x1, bbox_to_update.y1, 
                bbox_to_update.x2, bbox_to_update.y2,
                new_track_id=track_id
            )
            self.force_redraw_current_frame()
            self.update_active_tracks()
            self.update_ui_state()
            self.status_bar.showMessage(f"Assigned track ID {track_id}", 2000)

    def delete_selected_bbox(self):
        """Delete the currently selected bbox."""
        if self.selected_bbox_instance_id is None:
            self.status_bar.showMessage("No bbox selected to delete.", 2000)
            return
        
        self.bbox_manager.delete_bbox(self.video_decoder.current_frame, 
                                    self.selected_bbox_instance_id)
        self.selected_bbox_instance_id = None
        self.force_redraw_current_frame()
        self.update_ui_state()
        self.status_bar.showMessage("Bbox deleted.", 2000)

    def force_redraw_current_frame(self):
        """Force redraw of current frame."""
        self.video_decoder.force_refresh = True
        
    def update_active_tracks(self):
        """Update the display of active track IDs."""
        active_ids = sorted(list(self.bbox_manager.active_track_ids))
        self.active_tracks_label.setText(f"Active IDs: {', '.join(map(str, active_ids))}")

    def interpolate_selected_track(self):
        """Interpolate the selected track between keyframes."""
        if not self.selected_bbox_instance_id:
            self.status_bar.showMessage("Select a bbox first", 2000)
            return
        
        current_frame = self.video_decoder.current_frame
        bbox = next((b for b in self.bbox_manager.get_bboxes(current_frame) 
                     if b.instance_id == self.selected_bbox_instance_id), None)
        
        if not bbox or bbox.track_id is None:
            self.status_bar.showMessage("Selected bbox needs a track ID", 2000)
            return
        
        # Find previous and next keyframes with this track
        track_id = bbox.track_id
        prev_frame = self.find_prev_frame_with_track(track_id, current_frame)
        next_frame = self.find_next_frame_with_track(track_id, current_frame)
        
        if prev_frame is not None and next_frame is not None:
            self.bbox_manager.interpolate_track(track_id, prev_frame, next_frame)
            self.force_redraw_current_frame()
            self.update_ui_state()
            self.status_bar.showMessage(
                f"Interpolated track {track_id} between frames {prev_frame}-{next_frame}", 3000
            )
        else:
            self.status_bar.showMessage(f"Could not find both keyframes for track {track_id}", 2000)

    def find_prev_frame_with_track(self, track_id: int, current_frame: int) -> Optional[int]:
        """Find previous frame containing the specified track."""
        for frame in range(current_frame - 1, -1, -1):
            bboxes = self.bbox_manager.get_bboxes(frame)
            if any(b.track_id == track_id for b in bboxes):
                return frame
        return None

    def find_next_frame_with_track(self, track_id: int, current_frame: int) -> Optional[int]:
        """Find next frame containing the specified track."""
        for frame in range(current_frame + 1, self.video_decoder.total_frames):
            bboxes = self.bbox_manager.get_bboxes(frame)
            if any(b.track_id == track_id for b in bboxes):
                return frame
        return None

    def jump_to_prev_detection(self):
        """Jump to previous frame with detections."""
        current_frame = self.video_decoder.current_frame
        detection_frames = self.bbox_manager.get_frames_with_detections()
        
        prev_frames = [f for f in detection_frames if f < current_frame]
        if prev_frames:
            self.video_decoder.seek_to_frame(prev_frames[-1])
            self.status_bar.showMessage(f"Jumped to frame {prev_frames[-1]}", 1000)
        else:
            self.status_bar.showMessage("No previous detections", 1000)

    def jump_to_next_detection(self):
        """Jump to next frame with detections."""
        current_frame = self.video_decoder.current_frame
        detection_frames = self.bbox_manager.get_frames_with_detections()
        
        next_frames = [f for f in detection_frames if f > current_frame]
        if next_frames:
            self.video_decoder.seek_to_frame(next_frames[0])
            self.status_bar.showMessage(f"Jumped to frame {next_frames[0]}", 1000)
        else:
            self.status_bar.showMessage("No next detections", 1000)
    
    def update_confidence_threshold(self, value):
        """Update confidence threshold for filtering."""
        self.confidence_threshold = value / 100.0
        self.conf_label.setText(f"{self.confidence_threshold:.2f}")
        self.force_redraw_current_frame()
    
    def toggle_track_trails(self, enabled):
        """Toggle track trail visualization."""
        self.show_track_trails = enabled
        self.force_redraw_current_frame()
    
    def toggle_auto_save(self, enabled):
        """Toggle auto-save functionality."""
        self.auto_save_enabled = enabled
        if enabled:
            if self.bbox_manager.modified:
                self.autosave_timer.start(30000)  # 30 seconds
        else:
            self.autosave_timer.stop()
    
    def update_label_threshold(self, value):
        """Update FPS threshold for label drawing."""
        self.label_draw_threshold = value
    
    def auto_save(self):
        """Perform auto-save if enabled and modifications exist."""
        if self.bbox_manager.modified and self.auto_save_enabled:
            backup_path = self.bbox_manager.json_path + '.autosave'
            try:
                self.bbox_manager.save_json(backup_path)
                self.status_bar.showMessage("Auto-saved", 2000)
            except Exception as e:
                self.status_bar.showMessage(f"Auto-save failed: {e}", 3000)
    
    def get_track_trail(self, track_id: int, current_frame: int, num_frames: int = 30):
        """Get semi-transparent boxes from previous frames for track visualization."""
        trail_items = []
        for i in range(max(0, current_frame - num_frames), current_frame):
            bboxes = self.bbox_manager.get_bboxes(i)
            for bbox in bboxes:
                if bbox.track_id == track_id:
                    # Calculate opacity based on distance from current frame
                    opacity = 0.3 * (i - current_frame + num_frames) / num_frames
                    trail_items.append((bbox, opacity))
        return trail_items
    
    def is_track_suspicious(self, bbox: BBox) -> bool:
        """Check if a track shows suspicious behavior."""
        if bbox.track_id is None:
            return False
        
        current_frame = self.video_decoder.current_frame
        
        # Check for sudden position jumps
        prev_frame = self.find_prev_frame_with_track(bbox.track_id, current_frame)
        if prev_frame is not None and current_frame - prev_frame == 1:
            prev_box = next((b for b in self.bbox_manager.get_bboxes(prev_frame) 
                             if b.track_id == bbox.track_id), None)
            if prev_box:
                # Calculate center distance
                distance = bbox.distance_to(prev_box)
                
                # Calculate average box size for threshold
                avg_size = (bbox.width + bbox.height + prev_box.width + prev_box.height) / 4
                
                # Flag as suspicious if moved more than 2x the average box size
                if distance > avg_size * 2:
                    return True
        
        # Check for ID switches (multiple boxes with same ID in nearby frames)
        for offset in [-2, -1, 1, 2]:
            check_frame = current_frame + offset
            if 0 <= check_frame < self.video_decoder.total_frames:
                frame_boxes = self.bbox_manager.get_bboxes(check_frame)
                same_id_boxes = [b for b in frame_boxes if b.track_id == bbox.track_id]
                if len(same_id_boxes) > 1:
                    return True
        
        return False

    # Playback control methods
    def toggle_playback(self):
        """Toggle video playback."""
        if self.video_decoder.playing:
            self.video_decoder.pause()
            self.play_btn.setText("Play (Space)")
        else:
            self.video_decoder.play()
            self.play_btn.setText("Pause (Space)")

    def change_speed(self, speed_text: str):
        """Change playback speed."""
        speed = float(speed_text.rstrip('x'))
        self.video_decoder.set_speed(speed)

    def seek_from_slider(self, frame_num: int):
        """Handle timeline slider movement."""
        if self.timeline.isSliderDown():
            self.pending_seek_frame = frame_num
            if not self.scrub_timer.isActive():
                self.scrub_timer.start(50)
        else:
            self.video_decoder.seek_to_frame(frame_num)

    def perform_delayed_seek(self):
        """Perform delayed seek for smooth scrubbing."""
        if self.pending_seek_frame != -1:
            self.video_decoder.seek_to_frame(self.pending_seek_frame)
            self.pending_seek_frame = -1
    
    def on_slider_released(self):
        """Handle timeline slider release."""
        self.scrub_timer.stop()
        self.video_decoder.seek_to_frame(self.timeline.value())

    def prev_frame(self): 
        """Go to previous frame."""
        self.video_decoder.seek_to_frame(self.video_decoder.current_frame - 1)
        
    def next_frame(self): 
        """Go to next frame."""
        self.video_decoder.seek_to_frame(self.video_decoder.current_frame + 1)
    
    def prev_video(self): 
        """Load previous video in list."""
        self.load_video_pair(self.current_video_idx - 1)
        
    def next_video(self): 
        """Load next video in list."""
        self.load_video_pair(self.current_video_idx + 1)

    def toggle_downsampling(self, checked: bool):
        """Toggle video downsampling for performance."""
        self.downsampling = checked
        self.video_decoder.set_downsampling(checked)

    def save_changes(self):
        """Save current modifications to JSON file."""
        try:
            self.bbox_manager.save_json()
            self.status_bar.showMessage("Changes saved successfully!", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save changes: {e}")
        self.update_ui_state()
        
    def check_unsaved_changes(self):
        """Check for unsaved changes and prompt user."""
        if self.bbox_manager.modified:
            reply = QMessageBox.question(
                self, 'Unsaved Changes', 'Do you want to save your changes?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save: 
                self.save_changes()
            elif reply == QMessageBox.Cancel: 
                return False
        return True

    def closeEvent(self, event):
        """Handle application close event."""
        if self.check_unsaved_changes():
            self.video_decoder.stop_thread()
            self.video_decoder.wait(2000)
            event.accept()
        else:
            event.ignore()


def main():
    """Main entry point."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Video Tracking Quality Control GUI - Enhanced")
    parser.add_argument('--csv', required=True, 
                       help='CSV file with video_path,bbox_json_path columns')
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
        
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = TrackingQCWindow()
    window.load_csv(args.csv)
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()