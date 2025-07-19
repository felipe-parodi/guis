"""
Export utilities for different annotation formats.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .bbox_utils import BBoxManager


class COCOExporter:
    """Export tracking data to COCO format."""
    
    def __init__(self, bbox_manager: BBoxManager):
        self.bbox_manager = bbox_manager
        
    def export_to_coco(
        self,
        output_path: str,
        video_info: Dict,
        category_mapping: Optional[Dict[int, str]] = None,
        include_tracks: bool = True
    ) -> bool:
        """
        Export bounding boxes to COCO format.
        
        Args:
            output_path: Path to save COCO JSON file
            video_info: Video metadata (width, height, fps, etc.)
            category_mapping: Optional mapping of track_id to category name
            include_tracks: Whether to include track information
            
        Returns:
            True if export successful
        """
        try:
            coco_data = self._create_coco_structure(
                video_info, 
                category_mapping,
                include_tracks
            )
            
            with open(output_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"Exported COCO annotations to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to COCO format: {e}")
            return False
    
    def _create_coco_structure(
        self,
        video_info: Dict,
        category_mapping: Optional[Dict[int, str]],
        include_tracks: bool
    ) -> Dict:
        """Create COCO-format data structure."""
        
        # COCO info section
        info = {
            "description": "Video tracking annotations",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Video Tracking QC Tool",
            "date_created": datetime.now().isoformat()
        }
        
        # COCO images section (frames)
        images = []
        frames_with_data = self.bbox_manager.get_frames_with_detections()
        
        for frame_id in frames_with_data:
            images.append({
                "id": frame_id,
                "width": video_info.get('width', 1920),
                "height": video_info.get('height', 1080),
                "file_name": f"frame_{frame_id:06d}.jpg",
                "date_captured": None
            })
        
        # COCO categories section
        categories = self._create_categories(category_mapping)
        
        # COCO annotations section
        annotations = []
        annotation_id = 1
        
        for frame_id in frames_with_data:
            bboxes = self.bbox_manager.get_bboxes(frame_id)
            
            for bbox in bboxes:
                # Convert to COCO bbox format [x, y, width, height]
                coco_bbox = [
                    bbox.x1,
                    bbox.y1,
                    bbox.width,
                    bbox.height
                ]
                
                # Determine category
                category_id = 1  # Default category
                if category_mapping and bbox.track_id in category_mapping:
                    # Find category ID by name
                    category_name = category_mapping[bbox.track_id]
                    for cat in categories:
                        if cat['name'] == category_name:
                            category_id = cat['id']
                            break
                
                annotation = {
                    "id": annotation_id,
                    "image_id": frame_id,
                    "category_id": category_id,
                    "bbox": coco_bbox,
                    "area": bbox.area,
                    "iscrowd": 0,
                    "segmentation": []  # Empty for bbox-only annotations
                }
                
                # Add tracking information if requested
                if include_tracks and bbox.track_id is not None:
                    annotation["track_id"] = bbox.track_id
                    
                if bbox.confidence is not None:
                    annotation["score"] = bbox.confidence
                
                annotations.append(annotation)
                annotation_id += 1
        
        return {
            "info": info,
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "licenses": [],
            "videos": [{
                "id": 1,
                "name": video_info.get('name', 'video'),
                "width": video_info.get('width', 1920),
                "height": video_info.get('height', 1080),
                "length": len(frames_with_data),
                "fps": video_info.get('fps', 30.0)
            }] if include_tracks else []
        }
    
    def _create_categories(
        self, 
        category_mapping: Optional[Dict[int, str]]
    ) -> List[Dict]:
        """Create COCO categories from track information."""
        
        if category_mapping:
            # Create categories from mapping
            categories = []
            used_names = set()
            cat_id = 1
            
            for track_id, category_name in category_mapping.items():
                if category_name not in used_names:
                    categories.append({
                        "id": cat_id,
                        "name": category_name,
                        "supercategory": "object"
                    })
                    used_names.add(category_name)
                    cat_id += 1
        else:
            # Default single category
            categories = [{
                "id": 1,
                "name": "object",
                "supercategory": "thing"
            }]
        
        return categories


class YOLOExporter:
    """Export tracking data to YOLO format."""
    
    def __init__(self, bbox_manager: BBoxManager):
        self.bbox_manager = bbox_manager
    
    def export_to_yolo(
        self,
        output_dir: str,
        video_info: Dict,
        class_mapping: Optional[Dict[int, int]] = None
    ) -> bool:
        """
        Export to YOLO format (one .txt file per frame).
        
        Args:
            output_dir: Directory to save YOLO annotation files
            video_info: Video metadata
            class_mapping: Optional mapping of track_id to class_id
            
        Returns:
            True if export successful
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            width = video_info.get('width', 1920)
            height = video_info.get('height', 1080)
            
            frames_with_data = self.bbox_manager.get_frames_with_detections()
            
            for frame_id in frames_with_data:
                output_file = os.path.join(output_dir, f"frame_{frame_id:06d}.txt")
                
                with open(output_file, 'w') as f:
                    bboxes = self.bbox_manager.get_bboxes(frame_id)
                    
                    for bbox in bboxes:
                        # Convert to YOLO format (normalized center x, y, width, height)
                        center_x = (bbox.x1 + bbox.x2) / 2 / width
                        center_y = (bbox.y1 + bbox.y2) / 2 / height
                        norm_width = bbox.width / width
                        norm_height = bbox.height / height
                        
                        # Determine class
                        class_id = 0  # Default class
                        if class_mapping and bbox.track_id in class_mapping:
                            class_id = class_mapping[bbox.track_id]
                        
                        # Write YOLO line
                        confidence_str = ""
                        if bbox.confidence is not None:
                            confidence_str = f" {bbox.confidence:.4f}"
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} "
                               f"{norm_width:.6f} {norm_height:.6f}{confidence_str}\n")
            
            print(f"Exported YOLO annotations to {output_dir}")
            return True
            
        except Exception as e:
            print(f"Error exporting to YOLO format: {e}")
            return False


class CSVExporter:
    """Export tracking data to CSV format."""
    
    def __init__(self, bbox_manager: BBoxManager):
        self.bbox_manager = bbox_manager
    
    def export_to_csv(self, output_path: str) -> bool:
        """
        Export tracking data to CSV format.
        
        Format: frame_id,track_id,x1,y1,x2,y2,confidence
        
        Args:
            output_path: Path to save CSV file
            
        Returns:
            True if export successful
        """
        try:
            import csv
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'frame_id', 'track_id', 'x1', 'y1', 'x2', 'y2', 
                    'width', 'height', 'confidence'
                ])
                
                # Write data
                frames_with_data = self.bbox_manager.get_frames_with_detections()
                
                for frame_id in sorted(frames_with_data):
                    bboxes = self.bbox_manager.get_bboxes(frame_id)
                    
                    for bbox in bboxes:
                        writer.writerow([
                            frame_id,
                            bbox.track_id if bbox.track_id is not None else '',
                            bbox.x1,
                            bbox.y1,
                            bbox.x2,
                            bbox.y2,
                            bbox.width,
                            bbox.height,
                            bbox.confidence if bbox.confidence is not None else ''
                        ])
            
            print(f"Exported CSV annotations to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV format: {e}")
            return False


def export_tracking_data(
    bbox_manager: BBoxManager,
    output_path: str,
    format_type: str,
    video_info: Dict,
    **kwargs
) -> bool:
    """
    Generic export function that dispatches to specific exporters.
    
    Args:
        bbox_manager: BBoxManager instance
        output_path: Output file/directory path
        format_type: 'coco', 'yolo', or 'csv'
        video_info: Video metadata
        **kwargs: Additional arguments for specific exporters
        
    Returns:
        True if export successful
    """
    format_type = format_type.lower()
    
    if format_type == 'coco':
        exporter = COCOExporter(bbox_manager)
        return exporter.export_to_coco(output_path, video_info, **kwargs)
    
    elif format_type == 'yolo':
        exporter = YOLOExporter(bbox_manager)
        return exporter.export_to_yolo(output_path, video_info, **kwargs)
    
    elif format_type == 'csv':
        exporter = CSVExporter(bbox_manager)
        return exporter.export_to_csv(output_path)
    
    else:
        print(f"Unknown export format: {format_type}")
        return False