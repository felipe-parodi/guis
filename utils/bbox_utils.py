"""
Bounding box utilities for video tracking QC.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BBox:
    """Bounding box with track ID."""
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None
    confidence: Optional[float] = None
    selected: bool = False
    # Unique identifier for the object instance itself
    instance_id: int = field(default_factory=lambda: int(time.time() * 1e6))

    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def iou(self, other: 'BBox') -> float:
        """Calculate intersection over union with another bbox."""
        # Calculate intersection
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def distance_to(self, other: 'BBox') -> float:
        """Calculate center distance to another bbox."""
        c1 = self.center
        c2 = other.center
        return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5


class BBoxManager:
    """Manages bounding box data and track IDs."""
    
    def __init__(self):
        self.bbox_data: Dict[int, List[BBox]] = {}
        self.json_path = None
        self.modified = False
        self.active_track_ids = set()

    def load_json(self, json_path: str):
        """Load bounding boxes from JSON file with universal format parsing."""
        self.json_path = json_path
        self.bbox_data = {}
        self.active_track_ids = set()
        self.modified = False
        print(f"Loading JSON from: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Universal parser for different formats
            frames_data = []
            if isinstance(data, list):  # List of frame objects
                frames_data = data
            elif isinstance(data, dict):  # Could be a dict with various keys
                if 'frames' in data:
                    frames_data = data['frames']
                elif 'instance_info' in data:
                    frames_data = data['instance_info']
                else:  # Try to parse dict keys as frame numbers
                    for k, v in data.items():
                        if k.isdigit() and 'instances' in v:
                            v['frame_id'] = int(k)
                            frames_data.append(v)
            
            for frame_obj in frames_data:
                frame_id = frame_obj.get('frame_id', frame_obj.get('frame', -1))
                if frame_id == -1:
                    continue
                
                self.bbox_data[frame_id] = []
                for inst in frame_obj.get('instances', []):
                    bbox = inst.get('bbox')
                    if not bbox or len(bbox) < 4:
                        continue
                    
                    track_id = inst.get('track_id', inst.get('instance_id'))
                    if isinstance(track_id, dict):
                        track_id = track_id.get('track_id')
                    
                    confidence = inst.get('confidence', inst.get('score'))

                    new_bbox = BBox(
                        x1=float(bbox[0]), y1=float(bbox[1]),
                        x2=float(bbox[2]), y2=float(bbox[3]),
                        track_id=int(track_id) if track_id is not None else None,
                        confidence=float(confidence) if confidence is not None else None
                    )
                    self.bbox_data[frame_id].append(new_bbox)
                    if new_bbox.track_id is not None:
                        self.active_track_ids.add(new_bbox.track_id)
                        
            print(f"Successfully parsed {len(self.bbox_data)} frames with detections.")
        except Exception as e:
            print(f"Error loading or parsing JSON: {e}")
            raise

    def get_bboxes(self, frame_id: int) -> List[BBox]:
        """Get all bboxes for a specific frame."""
        return self.bbox_data.get(frame_id, [])

    def update_bbox(self, frame_id: int, bbox_instance_id: int, 
                    x1: float, y1: float, x2: float, y2: float,
                    new_track_id: Optional[int] = None):
        """Update a specific bbox's coordinates and optionally its track ID."""
        if frame_id in self.bbox_data:
            for bbox in self.bbox_data[frame_id]:
                if bbox.instance_id == bbox_instance_id:
                    bbox.x1, bbox.y1, bbox.x2, bbox.y2 = x1, y1, x2, y2
                    if new_track_id is not None:
                        bbox.track_id = new_track_id
                        self.active_track_ids.add(new_track_id)
                    self.modified = True
                    return

    def add_bbox(self, frame_id: int, new_bbox: BBox):
        """Add a new bbox to a frame."""
        if frame_id not in self.bbox_data:
            self.bbox_data[frame_id] = []
        self.bbox_data[frame_id].append(new_bbox)
        if new_bbox.track_id is not None:
            self.active_track_ids.add(new_bbox.track_id)
        self.modified = True

    def delete_bbox(self, frame_id: int, bbox_instance_id: int):
        """Delete a bbox from a frame."""
        if frame_id in self.bbox_data:
            self.bbox_data[frame_id] = [
                b for b in self.bbox_data[frame_id] 
                if b.instance_id != bbox_instance_id
            ]
            self.modified = True

    def interpolate_track(self, track_id: int, start_frame: int, end_frame: int):
        """Smart interpolation with motion estimation using smoothstep."""
        if start_frame >= end_frame:
            return

        start_box = next(
            (b for b in self.get_bboxes(start_frame) if b.track_id == track_id), 
            None
        )
        end_box = next(
            (b for b in self.get_bboxes(end_frame) if b.track_id == track_id), 
            None
        )

        if not start_box or not end_box:
            print(f"Interpolation failed: Could not find track {track_id} "
                  f"on both start and end frames.")
            return

        # Extract center and size for smoother motion
        start_center = start_box.center
        end_center = end_box.center
        start_size = (start_box.width, start_box.height)
        end_size = (end_box.width, end_box.height)

        total_frames = end_frame - start_frame
        for i in range(1, total_frames):
            current_frame = start_frame + i
            # Remove any existing box with this track_id on the interpolated frame
            self.bbox_data[current_frame] = [
                b for b in self.get_bboxes(current_frame) 
                if b.track_id != track_id
            ]
            
            # Use smoothstep for more natural motion
            t = i / total_frames
            t_smooth = t * t * (3.0 - 2.0 * t)  # Smoothstep function
            
            # Interpolate center with smoothstep, size linearly
            center = (
                start_center[0] + (end_center[0] - start_center[0]) * t_smooth,
                start_center[1] + (end_center[1] - start_center[1]) * t_smooth
            )
            size = (
                start_size[0] + (end_size[0] - start_size[0]) * t,
                start_size[1] + (end_size[1] - start_size[1]) * t
            )
            
            # Convert back to corner coordinates
            x1 = center[0] - size[0] / 2
            y1 = center[1] - size[1] / 2
            x2 = center[0] + size[0] / 2
            y2 = center[1] + size[1] / 2
            
            interp_box = BBox(x1, y1, x2, y2, track_id, confidence=0.99)
            self.add_bbox(current_frame, interp_box)
        
        self.modified = True
        print(f"Interpolated track {track_id} for {total_frames-1} frames "
              f"with smooth motion.")

    def save_json(self, save_path: Optional[str] = None):
        """Save bboxes to JSON file with backup."""
        if not self.modified:
            print("No changes to save.")
            return

        path = save_path or self.json_path
        if not path:
            raise ValueError("No save path specified.")
        
        backup_path = path + '.bak'
        if os.path.exists(path):
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(path, backup_path)
            print(f"Created backup: {backup_path}")

        # Reconstruct the JSON data from our internal BBox representation
        output_data = {'frames': []}
        sorted_frame_ids = sorted(self.bbox_data.keys())

        for frame_id in sorted_frame_ids:
            frame_obj = {'frame_id': frame_id, 'instances': []}
            for bbox in self.bbox_data[frame_id]:
                instance = {
                    'bbox': [bbox.x1, bbox.y1, bbox.x2, bbox.y2],
                    'track_id': bbox.track_id,
                    'confidence': bbox.confidence
                }
                frame_obj['instances'].append(instance)
            output_data['frames'].append(frame_obj)

        with open(path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        self.modified = False
        print(f"Saved changes to {path}")

    def get_frames_with_detections(self) -> List[int]:
        """Get sorted list of frames that have detections."""
        return sorted([fid for fid, bboxes in self.bbox_data.items() if bboxes])
    
    def get_track_frames(self, track_id: int) -> List[int]:
        """Get all frames where a specific track appears."""
        frames = []
        for frame_id, bboxes in self.bbox_data.items():
            if any(b.track_id == track_id for b in bboxes):
                frames.append(frame_id)
        return sorted(frames)
    
    def get_track_trajectory(self, track_id: int) -> List[Tuple[int, float, float]]:
        """Get trajectory of a track as list of (frame, x, y) tuples."""
        trajectory = []
        for frame_id in self.get_track_frames(track_id):
            bbox = next(
                (b for b in self.get_bboxes(frame_id) if b.track_id == track_id),
                None
            )
            if bbox:
                trajectory.append((frame_id, *bbox.center))
        return trajectory