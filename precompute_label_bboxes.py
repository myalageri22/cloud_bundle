#!/usr/bin/env python3
"""
Precompute bounding boxes for all label NIfTI files in data/processed.

For each label file, compute (zmin, zmax), (ymin, ymax), (xmin, xmax) of the
foreground (voxels > 0), and save the center + shape info to a JSON cache.

This enables deterministic label-centered crops during validation without
relying on MONAI's CropForegroundd (which has pickling issues).
"""

import json
from pathlib import Path
import numpy as np

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")


def compute_label_bbox(label_file: Path) -> dict:
    """
    Compute bounding box of foreground voxels in a label NIfTI file.
    
    Returns:
        dict with keys: 'filename', 'center', 'bbox_shape', 'bbox_coords', 'foreground_count'
    """
    try:
        img = nib.load(str(label_file))
        data = img.get_fdata()
        
        # Ensure 3D
        if data.ndim > 3:
            data = data[..., 0]
        
        # Find foreground voxels
        fg_idx = np.argwhere(data > 0)
        if fg_idx.size == 0:
            # No foreground
            return {
                'filename': label_file.name,
                'center': None,
                'bbox_shape': None,
                'bbox_coords': None,
                'foreground_count': 0,
            }
        
        # Compute bbox
        zmin, ymin, xmin = fg_idx.min(axis=0)
        zmax, ymax, xmax = fg_idx.max(axis=0)
        
        # Center in bbox
        center = [(zmin + zmax) / 2.0, (ymin + ymax) / 2.0, (xmin + xmax) / 2.0]
        
        # Shape of bbox
        shape = [int(zmax - zmin + 1), int(ymax - ymin + 1), int(xmax - xmin + 1)]
        
        coords = {
            'z': [int(zmin), int(zmax)],
            'y': [int(ymin), int(ymax)],
            'x': [int(xmin), int(xmax)],
        }
        
        return {
            'filename': label_file.name,
            'center': center,
            'bbox_shape': shape,
            'bbox_coords': coords,
            'foreground_count': int(fg_idx.shape[0]),
        }
    except Exception as e:
        print(f"Error processing {label_file.name}: {e}")
        return {
            'filename': label_file.name,
            'center': None,
            'bbox_shape': None,
            'bbox_coords': None,
            'foreground_count': 0,
            'error': str(e),
        }


def main():
    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "data" / "processed"
    
    if not processed_dir.exists():
        print(f"Error: {processed_dir} does not exist")
        return
    
    # Find all label files
    label_files = sorted(processed_dir.glob("*/*_seg.nii.gz"))
    
    if not label_files:
        print(f"No label files found in {processed_dir}")
        return
    
    print(f"Found {len(label_files)} label files. Computing bboxes...")
    
    bboxes = {}
    for label_file in label_files:
        bbox_info = compute_label_bbox(label_file)
        bboxes[label_file.name] = bbox_info
        
        fg_count = bbox_info.get('foreground_count', 0)
        center = bbox_info.get('center')
        if center:
            print(f"  {label_file.name}: center={[f'{c:.1f}' for c in center]}, "
                  f"fg_voxels={fg_count}")
        else:
            print(f"  {label_file.name}: NO FOREGROUND")
    
    # Save to JSON
    cache_file = project_root / "label_bboxes.json"
    with open(cache_file, 'w') as f:
        json.dump(bboxes, f, indent=2)
    
    print(f"\nSaved label bboxes to {cache_file}")


if __name__ == "__main__":
    main()
