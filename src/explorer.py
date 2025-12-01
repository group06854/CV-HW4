import math
import numpy as np
from typing import List, Optional, Dict, Any

def smooth_path(poses: List[dict], smoothness: float = 0.5) -> List[dict]:
    """
    Smooth camera path with smooth camera rotations
    """
    if len(poses) <= 2:
        return poses
    
    # Extract positions and directions
    camera_positions = np.array([p['camera_position'] for p in poses])
    look_at_points = np.array([p['look_at'] for p in poses])
    look_directions = look_at_points - camera_positions
    
    # Normalize look directions
    look_norms = np.linalg.norm(look_directions, axis=1, keepdims=True)
    look_norms[look_norms == 0] = 1.0  # avoid division by zero
    look_directions_norm = look_directions / look_norms
    
    # Calculate cumulative distance
    distances = np.sqrt(np.sum(np.diff(camera_positions, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_dist[-1]
    
    # Create uniform parameters
    original_params = cumulative_dist / total_length if total_length > 0 else np.linspace(0, 1, len(poses))
    num_output_points = len(poses)
    uniform_params = np.linspace(0, 1, num_output_points)
    
    smooth_camera_positions = []
    smooth_look_at_points = []
    
    for t in uniform_params:
        # Find indices for interpolation
        idx = np.searchsorted(original_params, t) - 1
        idx = max(0, min(idx, len(original_params) - 2))
        
        # Local parameter between reference points
        t_local = (t - original_params[idx]) / (original_params[idx + 1] - original_params[idx])
        t_local = np.clip(t_local, 0, 1)
        
        # Camera position interpolation (cubic smoothing)
        t_smooth = t_local * t_local * (3 - 2 * t_local)  # smoothstep function
        
        camera_pos = (1 - t_smooth) * camera_positions[idx] + t_smooth * camera_positions[idx + 1]
        smooth_camera_positions.append(camera_pos)
        
        # SPHERICAL interpolation of directions (manual SLERP implementation)
        dir1 = look_directions_norm[idx]
        dir2 = look_directions_norm[idx + 1]
        
        # Cosine of angle between directions
        dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        angle = math.acos(dot_product)
        
        if angle < 1e-6:
            # Directions are almost identical
            smooth_dir = dir1
        else:
            # Manual SLERP implementation
            sin_angle = math.sin(angle)
            weight1 = math.sin((1 - t_smooth) * angle) / sin_angle
            weight2 = math.sin(t_smooth * angle) / sin_angle
            smooth_dir = weight1 * dir1 + weight2 * dir2
        
        # Apply smoothed direction
        look_ahead_distance = 3.0
        smooth_look_at = camera_pos + smooth_dir * look_ahead_distance
        smooth_look_at_points.append(smooth_look_at)
    
    # Convert back to dictionaries
    smooth_poses = []
    for i in range(num_output_points):
        smooth_poses.append({
            'camera_position': smooth_camera_positions[i].tolist(),
            'look_at': smooth_look_at_points[i].tolist()
        })
    
    return smooth_poses

def generate_straight_path(start_pos: List[float], end_pos: List[float], num_points: int = 40, next_point: Optional[List[float]] = None) -> List[dict]:
    """
    Generate straight path with smooth turns
    next_point: next point after end_pos for smooth turn anticipation
    """
    poses = []
    
    start_arr = np.array(start_pos)
    end_arr = np.array(end_pos)
    
    # Calculate movement direction
    movement_vector = end_arr - start_arr
    movement_length = np.linalg.norm(movement_vector)
    
    if movement_length > 0:
        movement_direction = movement_vector / movement_length
    else:
        movement_direction = np.array([1.0, 0.0, 0.0])  # fallback
    
    # If there's a next point, calculate direction for smooth turn
    turn_direction = None
    if next_point is not None:
        next_arr = np.array(next_point)
        turn_vector = next_arr - end_arr
        turn_length = np.linalg.norm(turn_vector)
        if turn_length > 0:
            turn_direction = turn_vector / turn_length
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        
        # Smooth position interpolation
        t_smooth = t * t * (3 - 2 * t)  # smoothstep
        
        camera_pos = [
            start_pos[0] + t_smooth * (end_pos[0] - start_pos[0]),
            start_pos[1] + t_smooth * (end_pos[1] - start_pos[1]),
            start_pos[2] + t_smooth * (end_pos[2] - start_pos[2])
        ]
        
        # SMOOTH LOOK DIRECTION CHANGE
        look_ahead_distance = 3.0
        
        if turn_direction is not None:
            # Smoothly transition from movement to turn
            turn_weight = max(0, (t - 0.5) * 2)  # Start turning in second half
            turn_weight = turn_weight * turn_weight  # Quadratic smoothing
            
            # Interpolate between current direction and turn direction
            blended_direction = (1 - turn_weight) * movement_direction + turn_weight * turn_direction
            blended_direction = blended_direction / np.linalg.norm(blended_direction)
            
            look_at = [
                camera_pos[0] + blended_direction[0] * look_ahead_distance,
                camera_pos[1] + 1.0,
                camera_pos[2] + blended_direction[2] * look_ahead_distance
            ]
        else:
            # Without turn - look strictly in movement direction
            look_at = [
                camera_pos[0] + movement_direction[0] * look_ahead_distance,
                camera_pos[1] + 1.0,
                camera_pos[2] + movement_direction[2] * look_ahead_distance
            ]
        
        poses.append({
            'camera_position': camera_pos,
            'look_at': look_at
        })
    
    return poses

def generate_multi_point_path(waypoints: List[List[float]], points_per_segment: int = 20) -> List[dict]:
    """
    Generate path through multiple points with SMOOTH transitions between segments
    """
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints are required")
    
    all_poses = []
    
    # Generate segments with turn information
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]
        
        print(f"Creating segment {i+1}/{len(waypoints)-1}: {start_point} -> {end_point}")
        
        # Determine next point for smooth turn (if exists)
        next_point = waypoints[i + 2] if i < len(waypoints) - 2 else None
        
        segment_poses = generate_straight_path(
            start_pos=start_point,
            end_pos=end_point,
            num_points=points_per_segment,
            next_point=next_point
        )
        
        if i == 0:
            all_poses.extend(segment_poses)
        else:
            # Skip first point to avoid duplication
            all_poses.extend(segment_poses[1:])
    
    print(f"Generated path through {len(waypoints)} points, total {len(all_poses)} camera positions")
    return all_poses

def generate_elliptical_path(
    ellipse_center_xz: List[float] = [0.0, 0.0],  # Ellipse center in XZ plane
    radius_x: float = 2.0,                         # Radius along X axis (width)
    radius_z: float = 1.0,                         # Radius along Z axis (height)
    camera_height: float = 0.4,                    # Camera height (constant)
    num_points: int = 60,
    start_angle: float = 0.0,
    end_angle: float = 2 * math.pi
) -> List[dict]:
    """
    Generate elliptical path parallel to XZ plane, camera always looks INWARD
    """
    poses = []
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        angle = start_angle + t * (end_angle - start_angle)
        
        # Camera position on ellipse in XZ plane
        camera_x = ellipse_center_xz[0] + radius_x * math.cos(angle)
        camera_z = ellipse_center_xz[1] + radius_z * math.sin(angle)
        camera_y = camera_height  # Constant height
        
        # For correct look direction calculate ellipse normal
        # Derivatives of parametric ellipse equation
        dx_dt = -radius_x * math.sin(angle)  # derivative by x
        dz_dt = radius_z * math.cos(angle)   # derivative by z
        
        # Normal to ellipse (perpendicular to tangent)
        normal_x = -dz_dt
        normal_z = dx_dt
        
        # Normalize normal
        normal_length = math.sqrt(normal_x**2 + normal_z**2)
        if normal_length > 0:
            normal_x /= normal_length
            normal_z /= normal_length
        
        # Look-at point - offset by normal inside ellipse
        look_ahead_distance = 3  # distance for look-at point
        look_at_x = camera_x + normal_x * look_ahead_distance
        look_at_z = camera_z + normal_z * look_ahead_distance
        look_at_y = camera_height + 1  # SAME HEIGHT!
        
        poses.append({
            'camera_position': [camera_x, camera_y, camera_z],
            'look_at': [look_at_x, look_at_y, look_at_z]
        })
    
    print(f"Elliptical path: center={ellipse_center_xz}, radii X/Z={radius_x}/{radius_z}, height={camera_height}")
    return poses