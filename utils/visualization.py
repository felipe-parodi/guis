"""
Visualization utilities for tracking data.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QPen, QColor, QPainterPath, QBrush, QLinearGradient
from PyQt5.QtWidgets import QGraphicsPathItem, QGraphicsEllipseItem

from .bbox_utils import BBox, BBoxManager


class VisualizationManager:
    """Manages visualization overlays for tracking data."""
    
    def __init__(self, bbox_manager: BBoxManager):
        self.bbox_manager = bbox_manager
        self.show_trajectories = False
        self.show_heatmap = False
        self.trajectory_length = 50  # frames
        self.heatmap_resolution = 20  # grid size
        
    def get_track_trajectory_path(
        self, 
        track_id: int, 
        current_frame: int,
        scale: QPointF
    ) -> Optional[QGraphicsPathItem]:
        """
        Create a QPainterPath for track trajectory.
        
        Args:
            track_id: Track ID to visualize
            current_frame: Current frame number
            scale: Scale factor for coordinates
            
        Returns:
            QGraphicsPathItem with trajectory path
        """
        trajectory = self.bbox_manager.get_track_trajectory(track_id)
        if len(trajectory) < 2:
            return None
        
        # Filter to recent frames
        recent_trajectory = [
            (frame, x, y) for frame, x, y in trajectory 
            if frame <= current_frame and frame >= current_frame - self.trajectory_length
        ]
        
        if len(recent_trajectory) < 2:
            return None
        
        # Create path
        path = QPainterPath()
        first_point = True
        
        for frame, x, y in recent_trajectory:
            point = QPointF(x * scale.x(), y * scale.y())
            if first_point:
                path.moveTo(point)
                first_point = False
            else:
                path.lineTo(point)
        
        # Create path item with styling
        path_item = QGraphicsPathItem(path)
        
        # Color based on track ID
        colors = [
            QColor(255, 0, 0, 180),    # Red
            QColor(0, 255, 0, 180),    # Green
            QColor(0, 0, 255, 180),    # Blue
            QColor(255, 255, 0, 180),  # Yellow
            QColor(255, 0, 255, 180),  # Magenta
            QColor(0, 255, 255, 180),  # Cyan
        ]
        color = colors[track_id % len(colors)]
        
        pen = QPen(color, 2)
        pen.setStyle(1)  # SolidLine
        path_item.setPen(pen)
        
        return path_item
    
    def get_trajectory_points(
        self,
        track_id: int,
        current_frame: int,
        scale: QPointF
    ) -> List[QGraphicsEllipseItem]:
        """
        Create dots for trajectory points with fading opacity.
        
        Returns:
            List of QGraphicsEllipseItem for trajectory points
        """
        trajectory = self.bbox_manager.get_track_trajectory(track_id)
        points = []
        
        for frame, x, y in trajectory:
            if frame > current_frame or frame < current_frame - self.trajectory_length:
                continue
                
            # Calculate opacity based on age
            age = current_frame - frame
            opacity = max(0.2, 1.0 - (age / self.trajectory_length))
            
            # Create point
            radius = 3
            point = QGraphicsEllipseItem(
                x * scale.x() - radius,
                y * scale.y() - radius,
                radius * 2,
                radius * 2
            )
            
            # Style point
            color = QColor(255, 100, 100)
            color.setAlphaF(opacity)
            point.setBrush(QBrush(color))
            point.setPen(QPen(QColor(0, 0, 0, int(opacity * 255)), 1))
            
            points.append(point)
            
        return points
    
    def generate_heatmap(
        self,
        frame_range: Tuple[int, int],
        frame_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Generate a heatmap of object presence over time.
        
        Args:
            frame_range: (start_frame, end_frame)
            frame_size: (width, height) of video frames
            
        Returns:
            Heatmap as numpy array
        """
        start_frame, end_frame = frame_range
        width, height = frame_size
        
        # Create accumulator grid
        grid_w = width // self.heatmap_resolution
        grid_h = height // self.heatmap_resolution
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Accumulate presence data
        for frame_id in range(start_frame, end_frame + 1):
            bboxes = self.bbox_manager.get_bboxes(frame_id)
            for bbox in bboxes:
                # Convert bbox center to grid coordinates
                cx, cy = bbox.center
                gx = int(cx / self.heatmap_resolution)
                gy = int(cy / self.heatmap_resolution)
                
                # Add gaussian-like contribution
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < grid_w and 0 <= ny < grid_h:
                            weight = np.exp(-(dx*dx + dy*dy) / 4.0)
                            heatmap[ny, nx] += weight
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Upscale to frame size
        heatmap_full = cv2.resize(
            heatmap, 
            (width, height), 
            interpolation=cv2.INTER_CUBIC
        )
        
        # Convert to color
        heatmap_color = cv2.applyColorMap(
            (heatmap_full * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        return heatmap_color
    
    def create_heatmap_overlay(
        self,
        heatmap: np.ndarray,
        opacity: float = 0.5
    ) -> np.ndarray:
        """
        Create semi-transparent heatmap overlay.
        
        Args:
            heatmap: Heatmap image
            opacity: Overlay opacity (0-1)
            
        Returns:
            RGBA image for overlay
        """
        # Convert to RGBA
        if len(heatmap.shape) == 2:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGBA)
        elif heatmap.shape[2] == 3:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGBA)
        
        # Apply opacity
        heatmap[:, :, 3] = (heatmap[:, :, 3] * opacity).astype(np.uint8)
        
        return heatmap
    
    def get_movement_statistics(
        self, 
        track_id: int
    ) -> Dict[str, float]:
        """
        Calculate movement statistics for a track.
        
        Returns:
            Dictionary with statistics (total_distance, avg_speed, etc.)
        """
        trajectory = self.bbox_manager.get_track_trajectory(track_id)
        if len(trajectory) < 2:
            return {
                'total_distance': 0,
                'avg_speed': 0,
                'max_speed': 0,
                'total_frames': len(trajectory)
            }
        
        distances = []
        for i in range(1, len(trajectory)):
            f1, x1, y1 = trajectory[i-1]
            f2, x2, y2 = trajectory[i]
            
            # Calculate distance
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            
            # Calculate speed (pixels per frame)
            frame_diff = f2 - f1
            if frame_diff > 0:
                speed = dist / frame_diff
                distances.append((dist, speed))
        
        if not distances:
            return {
                'total_distance': 0,
                'avg_speed': 0,
                'max_speed': 0,
                'total_frames': len(trajectory)
            }
        
        total_dist = sum(d[0] for d in distances)
        speeds = [d[1] for d in distances]
        
        return {
            'total_distance': total_dist,
            'avg_speed': np.mean(speeds),
            'max_speed': max(speeds),
            'min_speed': min(speeds),
            'total_frames': len(trajectory),
            'active_frames': trajectory[-1][0] - trajectory[0][0] + 1
        }
    
    def get_track_density_map(
        self,
        track_ids: List[int],
        frame_size: Tuple[int, int],
        kernel_size: int = 21
    ) -> np.ndarray:
        """
        Create a density map showing where tracks spend most time.
        
        Args:
            track_ids: List of track IDs to include
            frame_size: (width, height) of video
            kernel_size: Gaussian kernel size for smoothing
            
        Returns:
            Density map as numpy array
        """
        width, height = frame_size
        density = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate all trajectory points
        for track_id in track_ids:
            trajectory = self.bbox_manager.get_track_trajectory(track_id)
            for _, x, y in trajectory:
                ix, iy = int(x), int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    density[iy, ix] += 1
        
        # Smooth with Gaussian kernel
        if kernel_size > 0:
            density = cv2.GaussianBlur(density, (kernel_size, kernel_size), 0)
        
        # Normalize
        if density.max() > 0:
            density = density / density.max()
        
        return density