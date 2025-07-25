"""
Short-term tracking utilities using OpenCV trackers.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from .bbox_utils import BBox


class ShortTermTracker:
    """
    Wrapper for OpenCV short-term trackers.
    Supports multiple tracker types for different use cases.
    """
    
    # Handle different OpenCV versions
    TRACKER_TYPES = {}
    
    # Try new API first (OpenCV 4.5.1+)
    if hasattr(cv2, 'TrackerCSRT_create'):
        TRACKER_TYPES = {
            'csrt': cv2.TrackerCSRT_create,        # Accurate but slower
            'kcf': cv2.TrackerKCF_create,          # Fast, good for translation
            'mosse': cv2.TrackerMOSSE_create,      # Very fast but less accurate
            'mil': cv2.TrackerMIL_create,          # Good for partial occlusions
        }
    else:
        # Fallback for newer OpenCV versions (4.5.2+)
        try:
            TRACKER_TYPES = {
                'csrt': cv2.legacy.TrackerCSRT_create,
                'kcf': cv2.legacy.TrackerKCF_create,
                'mosse': cv2.legacy.TrackerMOSSE_create,
                'mil': cv2.legacy.TrackerMIL_create,
            }
        except AttributeError:
            # Final fallback - use available trackers
            TRACKER_TYPES = {}
            if hasattr(cv2, 'TrackerCSRT'):
                TRACKER_TYPES['csrt'] = lambda: cv2.TrackerCSRT()
            if hasattr(cv2, 'TrackerKCF'):
                TRACKER_TYPES['kcf'] = lambda: cv2.TrackerKCF()
            if hasattr(cv2, 'TrackerMOSSE'):
                TRACKER_TYPES['mosse'] = lambda: cv2.TrackerMOSSE()
            if hasattr(cv2, 'TrackerMIL'):
                TRACKER_TYPES['mil'] = lambda: cv2.TrackerMIL()
            
            # If none work, provide basic fallback
            if not TRACKER_TYPES:
                TRACKER_TYPES = {
                    'kcf': lambda: None,  # Will be handled in __init__
                }
    
    def __init__(self, tracker_type: str = 'csrt'):
        """
        Initialize tracker.
        
        Args:
            tracker_type: Type of tracker ('csrt', 'kcf', 'mosse', 'mil')
        """
        # Check if any trackers are available
        if not self.TRACKER_TYPES:
            raise RuntimeError("No OpenCV trackers available. Please install opencv-contrib-python.")
            
        # Test if the requested tracker actually works
        if tracker_type not in self.TRACKER_TYPES:
            original_tracker = tracker_type
            available_trackers = list(self.TRACKER_TYPES.keys())
            if available_trackers:
                tracker_type = available_trackers[0]
                print(f"Warning: Requested tracker '{original_tracker}' not available. "
                      f"Using '{tracker_type}' instead.")
            else:
                raise ValueError(f"Unknown tracker type: {original_tracker}. "
                               f"Available: {list(self.TRACKER_TYPES.keys())}")
        
        # Test tracker creation before setting it
        working_tracker = None
        for test_type in [tracker_type] + [t for t in self.TRACKER_TYPES.keys() if t != tracker_type]:
            try:
                test_tracker = self.TRACKER_TYPES[test_type]()
                if test_tracker is not None:
                    working_tracker = test_type
                    if test_type != tracker_type:
                        print(f"Warning: '{tracker_type}' failed to initialize. Using '{test_type}' instead.")
                    break
            except Exception as e:
                print(f"Debug: Tracker '{test_type}' failed to create: {e}")
                continue
        
        if working_tracker is None:
            raise RuntimeError("No working OpenCV trackers found. All tracker types failed to initialize.")
        
        self.tracker_type = working_tracker
        self.tracker = None
        self.is_initialized = False
        
    def init(self, frame: np.ndarray, bbox: BBox) -> bool:
        """
        Initialize tracker with first frame and bounding box.
        
        Args:
            frame: BGR frame from video
            bbox: Initial bounding box
            
        Returns:
            True if initialization successful
        """
        try:
            # Create new tracker instance with error handling
            self.tracker = self.TRACKER_TYPES[self.tracker_type]()
            
            # Handle case where tracker creation failed
            if self.tracker is None:
                print(f"Warning: Failed to create {self.tracker_type} tracker")
                self.is_initialized = False
                return False
            
            # Validate bbox coordinates
            if bbox.width <= 0 or bbox.height <= 0:
                print(f"Warning: Invalid bbox dimensions: {bbox.width}x{bbox.height}")
                self.is_initialized = False
                return False
            
            # Convert bbox to OpenCV format (x, y, width, height)
            cv_bbox = (
                max(0, int(bbox.x1)), 
                max(0, int(bbox.y1)), 
                max(1, int(bbox.width)), 
                max(1, int(bbox.height))
            )
            
            # Ensure bbox is within frame boundaries
            h, w = frame.shape[:2]
            if cv_bbox[0] >= w or cv_bbox[1] >= h:
                print(f"Warning: Bbox outside frame boundaries: {cv_bbox} vs {w}x{h}")
                self.is_initialized = False
                return False
            
            # Initialize tracker with comprehensive error handling
            try:
                self.is_initialized = self.tracker.init(frame, cv_bbox)
                if not self.is_initialized:
                    print(f"Warning: Tracker '{self.tracker_type}' init() returned False")
            except cv2.error as e:
                print(f"Warning: OpenCV error during tracker initialization: {e}")
                self.is_initialized = False
            except Exception as e:
                print(f"Warning: Unexpected error during tracker initialization: {e}")
                self.is_initialized = False
                
        except Exception as e:
            print(f"Warning: Failed to create tracker '{self.tracker_type}': {e}")
            self.is_initialized = False
            
        return self.is_initialized
    
    def update(self, frame: np.ndarray) -> Optional[BBox]:
        """
        Update tracker with new frame.
        
        Args:
            frame: BGR frame from video
            
        Returns:
            Updated BBox if tracking successful, None otherwise
        """
        if not self.is_initialized or self.tracker is None:
            return None
            
        try:
            # Update tracker with error handling
            success, cv_bbox = self.tracker.update(frame)
            
            if not success:
                print(f"Debug: Tracker '{self.tracker_type}' update returned False")
                self.is_initialized = False
                return None
            
            # Validate returned bbox
            if len(cv_bbox) != 4:
                print(f"Warning: Invalid bbox format returned: {cv_bbox}")
                self.is_initialized = False
                return None
            
            x, y, w, h = cv_bbox
            
            # Validate bbox values
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                print(f"Warning: Invalid bbox values: x={x}, y={y}, w={w}, h={h}")
                self.is_initialized = False
                return None
            
            return BBox(
                x1=float(x),
                y1=float(y),
                x2=float(x + w),
                y2=float(y + h),
                confidence=0.95  # High confidence for manual tracking
            )
            
        except cv2.error as e:
            print(f"Warning: OpenCV error during tracking update: {e}")
            self.is_initialized = False
            return None
        except Exception as e:
            print(f"Warning: Unexpected error during tracking update: {e}")
            self.is_initialized = False
            return None
    
    def reset(self):
        """Reset tracker to uninitialized state."""
        self.tracker = None
        self.is_initialized = False


class MultiTracker:
    """
    Manages multiple trackers for different objects.
    Useful for tracking multiple objects simultaneously.
    """
    
    def __init__(self, tracker_type: str = 'csrt'):
        self.tracker_type = tracker_type
        self.trackers = {}  # track_id -> ShortTermTracker
        
    def add_tracker(self, track_id: int, frame: np.ndarray, bbox: BBox) -> bool:
        """Add a new tracker for a specific track ID."""
        tracker = ShortTermTracker(self.tracker_type)
        if tracker.init(frame, bbox):
            self.trackers[track_id] = tracker
            return True
        return False
    
    def update_all(self, frame: np.ndarray) -> Dict[int, BBox]:
        """
        Update all active trackers.
        
        Returns:
            Dictionary of track_id -> updated BBox
        """
        results = {}
        failed_trackers = []
        
        for track_id, tracker in self.trackers.items():
            bbox = tracker.update(frame)
            if bbox:
                bbox.track_id = track_id
                results[track_id] = bbox
            else:
                failed_trackers.append(track_id)
        
        # Remove failed trackers
        for track_id in failed_trackers:
            del self.trackers[track_id]
            
        return results
    
    def remove_tracker(self, track_id: int):
        """Remove a specific tracker."""
        if track_id in self.trackers:
            self.trackers[track_id].reset()
            del self.trackers[track_id]
    
    def clear(self):
        """Remove all trackers."""
        for tracker in self.trackers.values():
            tracker.reset()
        self.trackers.clear()
    
    def get_active_tracks(self) -> List[int]:
        """Get list of currently tracked IDs."""
        return list(self.trackers.keys())


def track_forward(
    frames: List[np.ndarray], 
    initial_bbox: BBox,
    tracker_type: str = 'csrt',
    max_frames: int = 30
) -> List[Optional[BBox]]:
    """
    Track an object forward through a sequence of frames.
    
    Args:
        frames: List of BGR frames
        initial_bbox: Starting bounding box
        tracker_type: Type of tracker to use
        max_frames: Maximum frames to track
        
    Returns:
        List of BBoxes (None for failed frames)
    """
    if not frames:
        return []
    
    tracker = ShortTermTracker(tracker_type)
    results = []
    
    # Initialize with first frame
    if tracker.init(frames[0], initial_bbox):
        results.append(initial_bbox)
    else:
        return []
    
    # Track through remaining frames
    for i, frame in enumerate(frames[1:max_frames], 1):
        bbox = tracker.update(frame)
        results.append(bbox)
        if bbox is None:
            break
    
    return results