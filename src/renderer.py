import subprocess
import numpy as np
import torch
import math
from typing import List, Tuple, Dict, Any
from gsplat import rasterization

from src.detector import ObjectTracker, draw_detections, save_detection_report

class GaussianData:
    def __init__(self, means, quats, scales, colors, xys=None, depths=None, radii=None, conics=None):
        self.means = means
        self.quats = quats
        self.scales = scales
        self.colors = colors
        self.xys = xys
        self.depths = depths
        self.radii = radii
        self.conics = conics
        self.real_opacities = None
    
    def prepare_for_gsplat(self) -> Dict[str, torch.Tensor]:
        """Prepare data for gsplat functions with correct [1, N, *] format"""
        if hasattr(self, 'real_opacities') and self.real_opacities is not None:
            opacities = self.real_opacities.unsqueeze(0).unsqueeze(-1)
        else:
            opacities = torch.ones((1, self.means.shape[0], 1), dtype=torch.float32)
        
        return {
            'means': self.means.unsqueeze(0),
            'scales': self.scales.unsqueeze(0),
            'quats': self.quats.unsqueeze(0),
            'colors': self.colors.unsqueeze(0),
            'opacities': opacities,
        }
        
    def validate_shapes(self) -> bool:
        """Check tensor shape consistency"""
        n_gaussians: int = self.means.shape[0]
        return (
            self.means.shape == (n_gaussians, 3) and
            self.quats.shape == (n_gaussians, 4) and
            self.scales.shape == (n_gaussians, 3) and
            self.colors.shape == (n_gaussians, 3)
        )
    
    def get_device(self) -> torch.device:
        """Get tensors device"""
        return self.means.device
    
    def to(self, device: torch.device) -> 'GaussianData':
        """Move all tensors to specified device"""
        return GaussianData(
            means=self.means.to(device),
            quats=self.quats.to(device),
            scales=self.scales.to(device),
            colors=self.colors.to(device),
            xys=self.xys.to(device) if self.xys is not None else None,
            depths=self.depths.to(device) if self.depths is not None else None,
            radii=self.radii.to(device) if self.radii is not None else None,
            conics=self.conics.to(device) if self.conics is not None else None,
        )

def create_view_matrix(camera_pos: List[float], look_at: List[float], up: List[float] = [0.0, 1.0, 0.0]) -> torch.Tensor:
    """Create WORLD-TO-CAMERA view matrix - returns [4, 4] format"""
    cam_pos = torch.tensor(camera_pos, dtype=torch.float32)
    look_at = torch.tensor(look_at, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    device = torch.device('cuda')
    
    forward = torch.nn.functional.normalize(look_at - cam_pos, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(up, forward), dim=0)
    up = torch.nn.functional.normalize(torch.linalg.cross(forward, right), dim=0)
    
    view_matrix = torch.eye(4, dtype=torch.float32).to(device)
    
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up  
    view_matrix[2, :3] = forward
    
    view_matrix[0, 3] = -torch.dot(cam_pos, right)
    view_matrix[1, 3] = -torch.dot(cam_pos, up)
    view_matrix[2, 3] = -torch.dot(cam_pos, forward)
    
    return view_matrix

def create_camera_parameters(
    image_size: Tuple[int, int] = (1920, 1080),
    camera_position: List[float] = [0.0, 0.0, 5.0],
    look_at: List[float] = [0.0, 0.0, 0.0],
    fov_degrees: float = 70.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create viewmat and K matrices for camera"""
    H, W = image_size
    device = torch.device('cuda')
    
    viewmat = create_view_matrix(camera_position, look_at)
    viewmat = viewmat.unsqueeze(0)
    
    fx = fy = W / (2 * math.tan(math.radians(fov_degrees) / 2))
    K = torch.tensor([
        [fx, 0, W/2],
        [0, fy, H/2], 
        [0,  0,   1]
    ], dtype=torch.float32).to(device)
    K = K.unsqueeze(0)
    viewmat = viewmat.to(device)
    
    return viewmat, K

def render_with_rasterization(
    gaussian_data: GaussianData,
    image_size: Tuple[int, int] = (1920, 1080),
    camera_position: List[float] = [0.0, 0.0, 5.0],
    look_at: List[float] = [0.0, 0.0, 0.0],
    fov_degrees: float = 60.0
) -> torch.Tensor:
    """Rendering using rasterization"""
    
    means = gaussian_data.means
    quats = gaussian_data.quats
    scales = gaussian_data.scales
    colors = gaussian_data.colors
    
    opacities = gaussian_data.real_opacities
    viewmats, Ks = create_camera_parameters(
        image_size=image_size,
        camera_position=camera_position,
        look_at=look_at,
        fov_degrees=fov_degrees
    )

    render_colors, render_alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=image_size[0],
        height=image_size[1],
        near_plane=0.01,
        far_plane=1e4,
        sh_degree=None,
        packed=True,
        render_mode='RGB'
    )
    
    return render_colors[0]

def create_video_from_poses(
    gaussian_data: GaussianData,
    poses: List[dict],
    output_path: str = "output_video.mp4",
    image_size: Tuple[int, int] = (1920, 1080),
    fov_degrees: float = 60.0,
    fps: int = 30,
    enable_object_detection: bool = True
):
    """Create video from camera pose list with object detection capability"""
    
    W, H = image_size
    
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24',
        '-s', f'{W}x{H}',
        '-r', str(fps),
        '-i', '-',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    
    print(f"Creating video: {output_path}")
    
    # Initialize tracker if object detection is enabled
    tracker = None
    if enable_object_detection:
        print("Object detection enabled...")
        tracker = ObjectTracker()
    
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    
    try:
        for i, pose in enumerate(poses):
            print(f"Frame {i+1}/{len(poses)}")
            
            frame = render_with_rasterization(
                gaussian_data=gaussian_data,
                image_size=image_size,
                camera_position=pose['camera_position'],
                look_at=pose['look_at'],
                fov_degrees=fov_degrees
            )
            
            frame_np = frame.detach().cpu().numpy()
            frame_np = np.clip(frame_np, 0.0, 1.0)
            frame_np = (frame_np * 255).astype(np.uint8)
            
            # OBJECT DETECTION
            if enable_object_detection and tracker:
                # Detection
                detections = tracker.detect_objects_2d(frame_np)
                current_objects = tracker.update_tracker(detections, i)
                
                # Draw bbox on frame
                frame_np = draw_detections(frame_np, current_objects)
            
            process.stdin.write(frame_np.tobytes())
        
        process.stdin.close()
        process.wait()
        
        # Save report if detection was enabled
        if enable_object_detection and tracker:
            save_detection_report(tracker, output_path)
        
        print(f"Video created: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        process.terminate()
        raise