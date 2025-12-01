import math
from typing import List, Tuple, Optional
import numpy as np

def create_view_matrix(camera_pos: List[float], look_at: List[float], up: List[float] = [0.0, 1.0, 0.0]) -> np.ndarray:
    """Create view matrix for camera"""
    cam_pos = np.array(camera_pos, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    forward = look_at - cam_pos
    forward_norm = forward / np.linalg.norm(forward)
    
    right = np.cross(up, forward_norm)
    right_norm = right / np.linalg.norm(right)
    
    up_norm = np.cross(forward_norm, right_norm)
    
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[0, :3] = right_norm
    view_matrix[1, :3] = up_norm
    view_matrix[2, :3] = forward_norm
    
    view_matrix[0, 3] = -np.dot(cam_pos, right_norm)
    view_matrix[1, 3] = -np.dot(cam_pos, up_norm)
    view_matrix[2, 3] = -np.dot(cam_pos, forward_norm)
    
    return view_matrix

def create_camera_intrinsics(
    image_size: Tuple[int, int] = (1920, 1080),
    fov_degrees: float = 70.0
) -> np.ndarray:
    """Create camera intrinsics matrix"""
    H, W = image_size
    
    fx = fy = W / (2 * math.tan(math.radians(fov_degrees) / 2))
    
    K = np.array([
        [fx, 0, W/2],
        [0, fy, H/2], 
        [0,  0,   1]
    ], dtype=np.float32)
    
    return K

def calculate_camera_path(
    start_pos: List[float],
    end_pos: List[float],
    num_frames: int = 100,
    look_ahead_distance: float = 3.0
) -> List[dict]:
    """Calculate camera path between two points"""
    path = []
    
    start = np.array(start_pos)
    end = np.array(end_pos)
    
    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        
        # Cubic interpolation for smoothness
        t_smooth = t * t * (3 - 2 * t)
        
        camera_pos = start + (end - start) * t_smooth
        
        # Calculate movement direction
        direction = end - start
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([1.0, 0.0, 0.0])
        
        # Look-at point slightly ahead in movement direction
        look_at = camera_pos + direction * look_ahead_distance
        
        path.append({
            'camera_position': camera_pos.tolist(),
            'look_at': look_at.tolist()
        })
    
    return path

def generate_grid_path(
    center: List[float] = [0.0, 0.0, 0.0],
    size: float = 10.0,
    resolution: int = 10,
    height: float = 2.0
) -> List[dict]:
    """Generate grid-based path"""
    path = []
    
    for i in range(resolution):
        for j in range(resolution):
            x = center[0] - size/2 + (size / (resolution - 1)) * i if resolution > 1 else center[0]
            z = center[2] - size/2 + (size / (resolution - 1)) * j if resolution > 1 else center[2]
            
            camera_pos = [x, height, z]
            look_at = [center[0], height, center[2]]
            
            path.append({
                'camera_position': camera_pos,
                'look_at': look_at
            })
    
    return path