import numpy as np
import torch
import math
from plyfile import PlyData, PlyElement
from typing import Optional, List, Dict, Any

from src.explorer import smooth_path, generate_multi_point_path, generate_elliptical_path
from src.renderer import GaussianData, create_video_from_poses
from src.detector import ObjectTracker

def unpack_position(packed: int, min_val: List[float], max_val: List[float]) -> List[float]:
    """Unpack position from uint32 to 3D coordinates - returns List[float] format"""
    x: float = (packed & 0x3FF) / 1023.0
    y: float = ((packed >> 10) & 0x3FF) / 1023.0
    z: float = ((packed >> 20) & 0xFFF) / 4095.0
    return [
        min_val[0] + (max_val[0] - min_val[0]) * x,
        min_val[1] + (max_val[1] - min_val[1]) * y,
        min_val[2] + (max_val[2] - min_val[2]) * z
    ]

def unpack_rotation(packed: int) -> List[float]:
    """Unpack rotation from uint32 to quaternion [w, x, y, z] - returns List[float] format"""
    w: float = (packed & 0x3FF) / 1023.0
    x: float = ((packed >> 10) & 0x3FF) / 1023.0
    y: float = ((packed >> 20) & 0x3FF) / 1023.0
    
    # Restore fourth component (as in original code)
    z: float = math.sqrt(max(0.0, 1.0 - w * w - x * x - y * y))
    return [w, x, y, z]

def unpack_scale(packed: int, min_val: List[float], max_val: List[float]) -> List[float]:
    """Unpack scale from uint32 to 3D scale - returns List[float] format"""
    x: float = (packed & 0x3FF) / 1023.0
    y: float = ((packed >> 10) & 0x3FF) / 1023.0
    z: float = ((packed >> 20) & 0xFFF) / 4095.0
    return [
        min_val[0] + (max_val[0] - min_val[0]) * x,
        min_val[1] + (max_val[1] - min_val[1]) * y,
        min_val[2] + (max_val[2] - min_val[2]) * z
    ]

def unpack_color(packed: int, min_val: List[float], max_val: List[float]) -> List[float]:
    """Unpack color from uint32 to RGB - returns List[float] format"""
    r: float = (packed & 0xFF) / 255.0
    g: float = ((packed >> 8) & 0xFF) / 255.0
    b: float = ((packed >> 16) & 0xFF) / 255.0
    return [
        min_val[0] + (max_val[0] - min_val[0]) * r,
        min_val[1] + (max_val[1] - min_val[1]) * g,
        min_val[2] + (max_val[2] - min_val[2]) * b
    ]

def load_supersplat_ply(filename: str, max_points: int = 1500000) -> GaussianData:
    """
    Load SuperSplat PLY file with data unpacking
    
    Args:
        filename: Path to PLY file
        
    Returns:
        GaussianData: Unpacked Gaussian data
    """
    # Load PLY file
    plydata: PlyData = PlyData.read(filename)
    
    # Get data with type checking
    chunks: Any = plydata['chunk']
    vertices: Any = plydata['vertex']
    
    chunk_count: int = len(chunks)
    vertex_count: int = len(vertices)
    
    print(f"Loaded: {chunk_count} chunks, {vertex_count} vertices")
    
    # Prepare arrays - will be converted to torch.Tensor [N, 3/4] format
    all_positions: List[List[float]] = []
    all_rotations: List[List[float]] = []
    all_scales: List[List[float]] = []
    all_colors: List[List[float]] = []
    
    # Process each chunk
    for chunk_idx, chunk in enumerate(chunks):
        # Min/max values for denormalization
        min_pos: List[float] = [chunk['min_x'], chunk['min_y'], chunk['min_z']]
        max_pos: List[float] = [chunk['max_x'], chunk['max_y'], chunk['max_z']]
        
        min_scale: List[float] = [chunk['min_scale_x'], chunk['min_scale_y'], chunk['min_scale_z']]
        max_scale: List[float] = [chunk['max_scale_x'], chunk['max_scale_y'], chunk['max_scale_z']]
        
        min_color: List[float] = [chunk['min_r'], chunk['min_g'], chunk['min_b']]
        max_color: List[float] = [chunk['max_r'], chunk['max_g'], chunk['max_b']]
        
        # Process 256 vertices in chunk (as in original code)
        for i in range(256):
            vertex_idx: int = chunk_idx * 256 + i
            if vertex_idx >= vertex_count:
                break
                
            vertex: Any = vertices[vertex_idx]
            
            # Unpack data - all functions return List[float] format
            position: List[float] = unpack_position(vertex['packed_position'], min_pos, max_pos)
            rotation: List[float] = unpack_rotation(vertex['packed_rotation'])
            scale: List[float] = unpack_scale(vertex['packed_scale'], min_scale, max_scale)
            color: List[float] = unpack_color(vertex['packed_color'], min_color, max_color)
            
            all_positions.append(position)
            all_rotations.append(rotation)
            all_scales.append(scale)
            all_colors.append(color)
        
        if (chunk_idx + 1) % 1000 == 0:
            print(f"Processed {chunk_idx + 1}/{chunk_count} chunks")
    
    device = torch.device('cuda')
    means_tensor: torch.Tensor = torch.tensor(all_positions, dtype=torch.float32).to(device)
    quats_tensor: torch.Tensor = torch.tensor(all_rotations, dtype=torch.float32).to(device)
    scales_tensor: torch.Tensor = torch.tensor(all_scales, dtype=torch.float32).to(device)
    colors_tensor: torch.Tensor = torch.tensor(all_colors, dtype=torch.float32).to(device)

    num_points = means_tensor.shape[0]
    if num_points > max_points:
        print(f"⚠️ Downsampling: {num_points} -> {max_points} points ({max_points/num_points*100:.1f}% kept)")
        
        # Random uniform sampling
        indices = torch.randperm(num_points, device=device)[:max_points]
        
        means_tensor = means_tensor[indices]
        quats_tensor = quats_tensor[indices]
        scales_tensor = scales_tensor[indices]
        colors_tensor = colors_tensor[indices]
    
    gaussian_data: GaussianData = GaussianData(
        means=means_tensor,
        quats=quats_tensor,
        scales=scales_tensor,
        colors=colors_tensor,
    )
    
    # Validate results - only check loaded data, not computed fields
    if not gaussian_data.validate_shapes():
        raise ValueError("Inconsistent tensor shapes after loading")
    
    return gaussian_data

def load_gaussians_from_3dgs_ply(path: str, max_points: Optional[int] = 4000000) -> GaussianData:
    """
    Load Gaussians from 3DGS PLY format compatible with GaussianData class
    """
    print(f"[3DGS] Loading: {path}")
    ply = PlyData.read(path)

    # Check format
    if "vertex" not in ply:
        raise RuntimeError("PLY has no 'vertex' element")
    
    v = ply["vertex"]
    names = v.data.dtype.names
    print(f"[3DGS] Vertex fields: {names}")

    # Required fields for 3DGS format
    required = ["x", "y", "z", "scale_0", "scale_1", "scale_2", 
                "rot_0", "rot_1", "rot_2", "rot_3", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
    
    missing = [f for f in required if f not in names]
    if missing:
        raise RuntimeError(f"3DGS PLY missing fields: {missing}")

    # --- Load data ---
    device = torch.device('cuda')
    
    # Positions [N, 3]
    x = np.asarray(v["x"], np.float32)
    y = np.asarray(v["y"], np.float32)  
    z = np.asarray(v["z"], np.float32)
    means = np.stack([x, y, z], axis=1)

    # Scales [N, 3]
    s0 = np.asarray(v["scale_0"], np.float32)
    s1 = np.asarray(v["scale_1"], np.float32) 
    s2 = np.asarray(v["scale_2"], np.float32)
    scales_raw = np.stack([s0, s1, s2], axis=1)

    # Opacity [N] - NEW: load real opacity values
    op_raw = np.asarray(v["opacity"], np.float32)

    # Heuristic for scales (log -> exp if negative)
    frac_neg_scale = (scales_raw < 0.0).mean()
    if frac_neg_scale > 0.1:
        scales = np.exp(scales_raw)
        print(f"[3DGS] Scales treated as log-stddev (neg_frac={frac_neg_scale:.3f})")
    else:
        scales = scales_raw
        print(f"[3DGS] Scales treated as linear (neg_frac={frac_neg_scale:.3f})")

    # Heuristic for opacities (logits -> sigmoid if outside [0,1]) - NEW
    op_min, op_max = float(op_raw.min()), float(op_raw.max())
    if op_min < 0.0 or op_max > 1.0:
        opacities = 1.0 / (1.0 + np.exp(-op_raw))
        print(f"[3DGS] Opacities treated as logits (range={op_min:.3f}/{op_max:.3f})")
    else:
        opacities = op_raw
        print(f"[3DGS] Opacities treated as already in [0,1] (range={op_min:.3f}/{op_max:.3f})")

    # Quaternions [N, 4] - normalize
    q0 = np.asarray(v["rot_0"], np.float32)
    q1 = np.asarray(v["rot_1"], np.float32)
    q2 = np.asarray(v["rot_2"], np.float32) 
    q3 = np.asarray(v["rot_3"], np.float32)
    quats = np.stack([q0, q1, q2, q3], axis=1)

    # Normalize quaternions
    norm = np.linalg.norm(quats, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    quats = quats / norm

    # Colors [N, 3] - convert spherical harmonics to RGB
    fdc0 = np.asarray(v["f_dc_0"], np.float32)
    fdc1 = np.asarray(v["f_dc_1"], np.float32)
    fdc2 = np.asarray(v["f_dc_2"], np.float32)
    f_dc = np.stack([fdc0, fdc1, fdc2], axis=1)

    SH_C0 = 0.28209479177387814  # spherical harmonics constant
    colors = 0.5 + SH_C0 * f_dc
    colors = np.clip(colors, 0.0, 1.0)

    N = means.shape[0]
    print(f"[3DGS] Loaded {N} points")

    # --- Downsampling ---
    if max_points is not None and N > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, size=max_points, replace=False)
        
        means = means[idx]
        quats = quats[idx] 
        scales = scales[idx]
        opacities = opacities[idx]  # NEW: downsample opacity too
        colors = colors[idx]
        
        print(f"[3DGS] Downsampled {N} -> {max_points} points")

    # --- Convert to torch tensors ---
    means_tensor = torch.tensor(means, dtype=torch.float32).to(device)
    quats_tensor = torch.tensor(quats, dtype=torch.float32).to(device)
    scales_tensor = torch.tensor(scales, dtype=torch.float32).to(device)
    colors_tensor = torch.tensor(colors, dtype=torch.float32).to(device)
    opacities_tensor = torch.tensor(opacities, dtype=torch.float32).to(device)

    # Create GaussianData
    gaussian_data = GaussianData(
        means=means_tensor,
        quats=quats_tensor,
        scales=scales_tensor,
        colors=colors_tensor,
    )

    # Add real opacities as attribute - NEW
    gaussian_data.real_opacities = opacities_tensor

    # Validate shapes
    if not gaussian_data.validate_shapes():
        raise ValueError("Inconsistent tensor shapes after 3DGS loading")

    print(f"[3DGS] Successfully loaded {len(gaussian_data.means)} points with real opacities")
    return gaussian_data

def create_circular_path_example():
    """Example of creating circular path"""
    try:
        # 1. Load data
        print("Loading PLY file...")
        gaussian_data = load_gaussians_from_3dgs_ply("inputs/outdoor-drone_open.ply")
        
        print(f"Loaded {len(gaussian_data.means)} Gaussians")
        
        # 2. Generate circular path
        print("Generating circular path...")
        # Ellipse with width 2 and height 1
        elliptical_poses = generate_elliptical_path(
            ellipse_center_xz=[0.0, 0.0],
            radius_x=1.8,    # width along X
            radius_z=0.8,    # height along Z  
            camera_height=0,
            num_points=400
        )
        
        print(f"Generated {len(elliptical_poses)} circular path points")
        
        # 3. Smooth path (for circle can reduce smoothing or skip)
        print("Smoothing path...")
        smooth_poses = smooth_path(elliptical_poses, smoothness=0.3)  # Less smoothing for circle
        
        print(f"After smoothing: {len(smooth_poses)} points")
        
        # 4. Create video
        print("Creating video...")
        create_video_from_poses(
            gaussian_data=gaussian_data,
            poses=smooth_poses,
            output_path="outputs/circular_walkthrough.mp4",
            image_size=(1280, 720),
            fov_degrees=80.0,  # Larger FOV for better view in circular motion
            fps=30,  # Smoother video for circular motion
            enable_object_detection=True 
        )
        
        print("Done! Video saved as 'outputs/circular_walkthrough.mp4'")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def create_multi_point_path_example():
    """Example of creating multi-point path"""
    try:
        # 1. Load data
        print("Loading PLY file...")
        gaussian_data = load_gaussians_from_3dgs_ply("inputs/Museume_open.ply")
        
        print(f"Loaded {len(gaussian_data.means)} Gaussians")
        
        # 2. Define waypoints
        waypoints = [
            [0,0,0],
            [-3.8, 0, -0.25],
            [-9.5, 0, -1],
            [-10.3, 0, 1],
            [-10.3, 0, 2],
            [-7, 0, 2],
            [0,0,2],
            [3, 0, 1.5], 
            [10, 0, 1.5],
            [13, 0,0.3],
            [16, 0, -0.3],
            [19,0,-4],
            [28, 0, -4],
            [32, 0, -4.5],
            [38, 0, -5],
            [40,0,-4],
            [40, 0,0],
            [37, 0, 1],
            [20, 0, 1],
            [18, 0, 1.5],
            [16, 0, 0],
            [12, 0, 0],
            [8, 0, -2],
            [6, 0, -5],
            [0, -2.8, -5],
            [-4, -2.8, -5.5],
            [-5, -2.8, -4.5],
            [-9, -2.8, -4.5],
            [-10, -2.8, -2],
            [-11, -2.8, -2],
            [-11, -2.8, -1],
            [-10, -3, -1]
        ]
        
        print(f"Waypoints: {len(waypoints)} points")
        
        # 3. Generate multi-point path
        print("Generating multi-point path...")
        multi_poses = generate_multi_point_path(
            waypoints=waypoints,
            points_per_segment=200
        )
        
        print(f"Generated {len(multi_poses)} path points")
        
        # 4. Smooth path for smoothness
        print("Smoothing path...")
        smooth_poses = smooth_path(multi_poses, smoothness=0.7)
        
        print(f"After smoothing: {len(smooth_poses)} points")
        
        # 5. Create video
        print("Creating video...")
        create_video_from_poses(
            gaussian_data=gaussian_data,
            poses=smooth_poses,
            output_path="outputs/multi_point_walkthrough.mp4",
            image_size=(1280, 720),
            fov_degrees=120.0,
            fps=30,
            enable_object_detection=True 
        )
        
        print("Done! Video saved as 'outputs/multi_point_walkthrough.mp4'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage
    create_circular_path_example()
    create_multi_point_path_example()