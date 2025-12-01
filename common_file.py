#if something go wrang start this file in docker
import numpy as np
import torch
from plyfile import PlyData, PlyElement
import math
from typing import Dict, List, Tuple, Any, Optional, Union
from gsplat import rasterization
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Any

class ObjectTracker:
    """Трекинг объектов между кадрами (только conf > 0.65)"""
    
    def __init__(self, model_size: str = 'n'):
        self.detected_objects = {}
        self.next_object_id = 0
        self.confidence_threshold = 0.65  # порог уверенности
        
        # Загрузка YOLO модели
        model_path = f'yolov8{model_size}.pt'
        print(f"Загружаем YOLO модель: {model_path}")
        self.yolo_model = YOLO(model_path)
        
        print(f"Сохранение объектов с уверенностью > {self.confidence_threshold}")
    
    def detect_objects_2d(self, frame_np: np.ndarray) -> List[Dict]:
        """Детекция объектов в кадре с фильтрацией по уверенности"""
        try:
            results = self.yolo_model(frame_np, verbose=False)
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = self.yolo_model.names[class_id]
                        
                        # ФИЛЬТРАЦИЯ: только объекты с уверенностью > 0.65
                        if confidence > 0.65:
                            detections.append({
                                'bbox': (float(x1), float(y1), float(x2), float(y2)),
                                'class_name': class_name,
                                'confidence': confidence,
                                'class_id': class_id
                            })
            
            print(f"Обнаружено объектов (conf > 0.65): {len(detections)}")
            return detections
            
        except Exception as e:
            print(f"Ошибка детекции: {e}")
            return []    

    def update_tracker(self, detections: List[Dict], frame_number: int) -> Dict:
        """Обновляет трекер и возвращает объекты для текущего кадра"""
        current_frame_objects = {}
        
        for detection in detections:
            object_id = self._assign_object_id(detection)
            
            if object_id not in self.detected_objects:
                self.detected_objects[object_id] = {
                    'object_id': object_id,
                    'class_name': detection['class_name'],
                    'first_seen': frame_number,
                    'detection_count': 1,
                    'max_confidence': detection['confidence'],
                    'last_seen': frame_number,
                    'all_detections': [detection]
                }
            else:
                self.detected_objects[object_id]['detection_count'] += 1
                self.detected_objects[object_id]['last_seen'] = frame_number
                self.detected_objects[object_id]['all_detections'].append(detection)
                self.detected_objects[object_id]['max_confidence'] = max(
                    self.detected_objects[object_id]['max_confidence'], 
                    detection['confidence']
                )
            
            current_frame_objects[object_id] = self.detected_objects[object_id]
            current_frame_objects[object_id]['current_detection'] = detection
        
        return current_frame_objects
    
    def _assign_object_id(self, detection: Dict) -> int:
        """Назначает ID объекту (упрощенная версия без сложного трекинга)"""
        # Простая логика: каждый класс получает свои ID
        class_name = detection['class_name']
        
        # Ищем существующий объект того же класса в nearby позиции
        for obj_id, obj_data in self.detected_objects.items():
            if (obj_data['class_name'] == class_name and 
                obj_data['last_seen'] >= len(self.detected_objects) - 10):  # Недавно видели
                return obj_id
        
        # Новый объект
        new_id = self.next_object_id
        self.next_object_id += 1
        return new_id
    
    def get_summary_report(self) -> Dict:
        """Генерирует итоговый отчет"""
        summary = {}
        for obj_id, obj_data in self.detected_objects.items():
            class_name = obj_data['class_name']
            if class_name not in summary:
                summary[class_name] = []
            
            summary[class_name].append({
                'object_id': obj_id,
                'detection_count': obj_data['detection_count'],
                'max_confidence': obj_data['max_confidence'],
                'first_seen_frame': obj_data['first_seen'],
                'last_seen_frame': obj_data['last_seen']
            })
        
        return summary

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
        self.real_opacities = None  # Добавляем атрибут для реальных opacity
    
    def prepare_for_gsplat(self) -> Dict[str, torch.Tensor]:
        """Prepare data for gsplat functions with correct [1, N, *] format"""
        # Используем реальные opacities если они есть, иначе фиктивные
        if hasattr(self, 'real_opacities') and self.real_opacities is not None:
            opacities = self.real_opacities.unsqueeze(0).unsqueeze(-1)  # [1, N, 1]
        else:
            opacities = torch.ones((1, self.means.shape[0], 1), dtype=torch.float32)
        
        return {
            'means': self.means.unsqueeze(0),      # [1, N, 3]
            'scales': self.scales.unsqueeze(0),    # [1, N, 3]
            'quats': self.quats.unsqueeze(0),      # [1, N, 4]
            'colors': self.colors.unsqueeze(0),    # [1, N, 3]
            'opacities': opacities,                # [1, N, 1] - РЕАЛЬНЫЕ данные!
        }
        
    def validate_shapes(self) -> bool:
        """Check tensor shape consistency - only validate loaded data, not computed fields"""
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
    
    def prepare_for_gsplat(self) -> Dict[str, torch.Tensor]:
        """Prepare data for gsplat functions with correct [1, N, *] format"""
        return {
            'means': self.means.unsqueeze(0),      # [1, N, 3] format for gsplat
            'scales': self.scales.unsqueeze(0),    # [1, N, 3] format for gsplat
            'quats': self.quats.unsqueeze(0),      # [1, N, 4] format for gsplat
            'colors': self.colors.unsqueeze(0),    # [1, N, 3] format for gsplat
            'opacities': torch.ones((1, self.means.shape[0], 1), dtype=torch.float32),  # [1, N, 1] format for gsplat
        }



def draw_detections(frame_np: np.ndarray, current_objects: Dict) -> np.ndarray:
    """Рисует bbox только для объектов с уверенностью > 0.65"""
    annotated_frame = frame_np.copy()
    
    # Фиксированный набор цветов для стабильности
    colors = [
        (0, 255, 0),    # Зеленый
        (255, 0, 0),    # Синий
        (0, 0, 255),    # Красный
        (255, 255, 0),  # Голубой
        (255, 0, 255),  # Розовый
        (0, 255, 255),  # Желтый
        (128, 0, 128),  # Фиолетовый
        (255, 165, 0),  # Оранжевый
        (0, 128, 128),  # Бирюзовый
        (128, 128, 0),  # Оливковый
    ]
    
    for obj_id, obj_data in current_objects.items():
        if 'current_detection' not in obj_data:
            continue
            
        detection = obj_data['current_detection']
        
        # ПРОВЕРКА УВЕРЕННОСТИ: рисуем только если > 0.65
        if detection['confidence'] <= 0.65:
            continue
            
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        # Простой выбор цвета по ID объекта
        color = colors[obj_id % len(colors)]
        
        # Рисуем bbox
        cv2.rectangle(annotated_frame, 
                     (int(x1), int(y1)), (int(x2), int(y2)), 
                     color, 2)
        
        # Подпись с классом, ID и уверенностью
        label = f"{class_name} ID:{obj_id} ({confidence:.2f})"
        
        # Фон для текста
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, 
                     (int(x1), int(y1) - text_size[1] - 10),
                     (int(x1) + text_size[0], int(y1)),
                     color, -1)
        
        # Текст
        cv2.putText(annotated_frame, label, 
                   (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return annotated_frame

def save_detection_report(tracker: ObjectTracker, video_path: str):
    """Сохраняет отчет по объектам с conf > 0.65 в JSON файл"""
    report = tracker.get_summary_report()
    report_path = video_path.replace('.mp4', '_detection_report.json')
    
    # Добавляем мета-информацию в отчет
    full_report = {
        'metadata': {
            'total_unique_objects': sum(len(objects) for objects in report.values()),
            'total_classes': len(report),
            'confidence_threshold': 0.65,
            'video_source': video_path
        },
        'objects_by_class': report
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    
    print(f"Отчет сохранен: {report_path}")
    
    # Вывод статистики в консоль
    print("\n=== СТАТИСТИКА ОБНАРУЖЕНИЯ (conf > 0.65) ===")
    total_objects = 0
    total_detections = 0
    
    for class_name, objects in report.items():
        class_detections = sum(obj['detection_count'] for obj in objects)
        total_detections += class_detections
        total_objects += len(objects)
        
        print(f"\n{class_name}:")
        print(f"  Уникальных объектов: {len(objects)}")
        print(f"  Всего обнаружений: {class_detections}")
        
        for obj in objects:
            print(f"    ID {obj['object_id']}: {obj['detection_count']} обнаружений, макс. уверенность: {obj['max_confidence']:.3f}")
    
    print(f"\nИтого (conf > 0.65):")
    print(f"  Всего уникальных объектов: {total_objects}")
    print(f"  Всего классов: {len(report)}")
    print(f"  Всего обнаружений: {total_detections}")

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
        GaussianData: Unpacked Gaussian data with:
            - means: torch.Tensor [N, 3] format - WILL be unsqueezed to [1, N, 3] for gsplat
            - quats: torch.Tensor [N, 4] format - WILL be unsqueezed to [1, N, 4] for gsplat  
            - scales: torch.Tensor [N, 3] format - WILL be unsqueezed to [1, N, 3] for gsplat
            - colors: torch.Tensor [N, 3] format - WILL be unsqueezed to [1, N, 3] for gsplat
            - xys: None (will be computed by gsplat) - WILL be unsqueezed to [1, N, 2] for gsplat
            - depths: None (will be computed by gsplat) - WILL be unsqueezed to [1, N, 1] for gsplat
            - radii: None (will be computed by gsplat) - WILL be unsqueezed to [1, N, 1] for gsplat 
            - conics: None (will be computed by gsplat) - WILL be unsqueezed to [1, N, 3] for gsplat
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
    # Convert to torch.Tensor with strict formats:
    means_tensor: torch.Tensor = torch.tensor(all_positions, dtype=torch.float32).to(device)      # [N, 3] format
    quats_tensor: torch.Tensor = torch.tensor(all_rotations, dtype=torch.float32).to(device)      # [N, 4] format
    scales_tensor: torch.Tensor = torch.tensor(all_scales, dtype=torch.float32).to(device)        # [N, 3] format
    colors_tensor: torch.Tensor = torch.tensor(all_colors, dtype=torch.float32).to(device)        # [N, 3] format

    num_points = means_tensor.shape[0]
    if num_points > max_points:
        print(f"⚠️ Downsampling: {num_points} -> {max_points} points ({max_points/num_points*100:.1f}% kept)")
        
        # Случайный равномерный отбор
        indices = torch.randperm(num_points, device=device)[:max_points]
        
        means_tensor = means_tensor[indices]
        quats_tensor = quats_tensor[indices]
        scales_tensor = scales_tensor[indices]
        colors_tensor = colors_tensor[indices]
    
    gaussian_data: GaussianData = GaussianData(
        means=means_tensor,    # [N, 3] format - WILL be unsqueezed to [1, N, 3] for gsplat
        quats=quats_tensor,    # [N, 4] format - WILL be unsqueezed to [1, N, 4] for gsplat
        scales=scales_tensor,  # [N, 3] format - WILL be unsqueezed to [1, N, 3] for gsplat
        colors=colors_tensor,  # [N, 3] format - WILL be unsqueezed to [1, N, 3] for gsplat
        # xys, depths, radii, conics remain None as required - WILL be unsqueezed when computed
    )
    
    # Validate results - only check loaded data, not computed fields
    if not gaussian_data.validate_shapes():
        raise ValueError("Inconsistent tensor shapes after loading")
    
    return gaussian_data


def create_view_matrix(camera_pos: List[float], look_at: List[float], up: List[float] = [0.0, 1.0, 0.0]) -> torch.Tensor:
    """Create WORLD-TO-CAMERA view matrix - returns [4, 4] format"""
    cam_pos = torch.tensor(camera_pos, dtype=torch.float32)
    look_at = torch.tensor(look_at, dtype=torch.float32)
    up = torch.tensor(up, dtype=torch.float32)
    device = torch.device('cuda')
    
    # Calculate camera basis
    forward = torch.nn.functional.normalize(look_at - cam_pos, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(up, forward), dim=0)  # Исправлено предупреждение
    up = torch.nn.functional.normalize(torch.linalg.cross(forward, right), dim=0)  # Исправлено предупреждение
    
    # Create WORLD-TO-CAMERA view matrix
    view_matrix = torch.eye(4, dtype=torch.float32).to(device)
    
    # Заполняем ПОВЕРНУТУЮ и ТРАНСПОНИРОВАННУЮ матрицу
    view_matrix[0, :3] = right      # X axis (right) - СТРОКА
    view_matrix[1, :3] = up         # Y axis (up) - СТРОКА  
    view_matrix[2, :3] = forward    # Z axis (forward) - СТРОКА
    
    # Translation: -dot(position, axis)
    view_matrix[0, 3] = -torch.dot(cam_pos, right)    # -pos·right
    view_matrix[1, 3] = -torch.dot(cam_pos, up)       # -pos·up  
    view_matrix[2, 3] = -torch.dot(cam_pos, forward)  # -pos·forward
    
    return view_matrix

def create_camera_parameters(
    image_size: Tuple[int, int] = (1920, 1080),
    camera_position: List[float] = [0.0, 0.0, 5.0],
    look_at: List[float] = [0.0, 0.0, 0.0],
    fov_degrees: float = 70.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Создает viewmat и K матрицы для камеры"""
    H, W = image_size
    device = torch.device('cuda')
    # 1. Создаем viewmat [4, 4]
    viewmat = create_view_matrix(camera_position, look_at)  # [4, 4]
    viewmat = viewmat.unsqueeze(0)  # [1, 4, 4] для batch
    
    # 2. Создаем K матрицу интринсиков [3, 3]
    fx = fy = W / (2 * math.tan(math.radians(fov_degrees) / 2))
    K = torch.tensor([
        [fx, 0, W/2],
        [0, fy, H/2], 
        [0,  0,   1]
    ], dtype=torch.float32).to(device)
    K = K.unsqueeze(0)  # [1, 3, 3] для batch
    viewmat = viewmat.to(device)
    return viewmat, K

def create_opacities(n_gaussians: int) -> torch.Tensor:
    """Создает тензор непрозрачностей"""
    device = torch.device('cuda')
    return torch.ones(n_gaussians, dtype=torch.float32).to(device)  # [N]

def render_with_rasterization(
    gaussian_data: GaussianData,
    image_size: Tuple[int, int] = (1920, 1080),
    camera_position: List[float] = [0.0, 0.0, 5.0],
    look_at: List[float] = [0.0, 0.0, 0.0],
    fov_degrees: float = 60.0
) -> torch.Tensor:
    """Рендеринг с использованием rasterization"""
    
    # 1. Получаем данные которые УЖЕ ЕСТЬ
    means = gaussian_data.means      # [N, 3] ✅
    quats = gaussian_data.quats      # [N, 4] ✅
    scales = gaussian_data.scales    # [N, 3] ✅
    colors = gaussian_data.colors    # [N, 3] ✅
    
    # 2. Создаем данные которых НЕТ
    opacities =  gaussian_data.real_opacities# [N] ✅ создаем
    viewmats, Ks = create_camera_parameters(      # [1,4,4] [1,3,3] ✅ создаем
        image_size=image_size,
        camera_position=camera_position,
        look_at=look_at,
        fov_degrees=fov_degrees
    )

    # 3. Вызываем rasterization
    render_colors, render_alphas, meta = rasterization(
        means=means,        # [N, 3] ✅
        quats=quats,        # [N, 4] ✅  
        scales=scales,      # [N, 3] ✅
        opacities=opacities,# [N] ✅
        colors=colors,      # [N, 3] ✅
        viewmats=viewmats,  # [1, 4, 4] ✅
        Ks=Ks,             # [1, 3, 3] ✅
        width=image_size[0],# int ✅
        height=image_size[1],# int ✅
        near_plane=0.01,
        far_plane=1e4,
        sh_degree=None,
        packed=True,
        render_mode='RGB'   # рендерим только цвета
    )
    
    # render_colors имеет shape [1, H, W, 3]
    return render_colors[0]  # убираем batch dimension → [H, W, 3]

def create_video_from_poses(
    gaussian_data: GaussianData,
    poses: List[dict],
    output_path: str = "output_video.mp4",
    image_size: Tuple[int, int] = (1920, 1080),
    fov_degrees: float = 60.0,
    fps: int = 30,
    enable_object_detection: bool = True  # Новая опция!
):
    """Создает видео из списка поз камеры с возможностью детекции объектов"""
    
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
    
    print(f"Создаем видео: {output_path}")
    
    # Инициализация трекера если включена детекция
    tracker = None
    if enable_object_detection:
        print("Включена детекция объектов...")
        tracker = ObjectTracker()
    
    process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    
    try:
        for i, pose in enumerate(poses):
            print(f"Кадр {i+1}/{len(poses)}")
            
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
            
            # ДЕТЕКЦИЯ ОБЪЕКТОВ
            if enable_object_detection and tracker:
                # Детекция
                detections = tracker.detect_objects_2d(frame_np)
                current_objects = tracker.update_tracker(detections, i)
                
                # Рисуем bbox поверх кадра
                frame_np = draw_detections(frame_np, current_objects)
            
            process.stdin.write(frame_np.tobytes())
        
        process.stdin.close()
        process.wait()
        
        # Сохраняем отчет если была детекция
        if enable_object_detection and tracker:
            save_detection_report(tracker, output_path)
        
        print(f"Видео создано: {output_path}")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        process.terminate()
        raise

def load_gaussians_from_3dgs_ply(path: str, max_points: Optional[int] = 4000000) -> 'GaussianData':
    """
    Load Gaussians from 3DGS PLY format compatible with your existing GaussianData class
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

    # Opacity [N] - НОВОЕ: загрузка реальных opacity
    op_raw = np.asarray(v["opacity"], np.float32)

    # Heuristic for scales (log -> exp if negative)
    frac_neg_scale = (scales_raw < 0.0).mean()
    if frac_neg_scale > 0.1:
        scales = np.exp(scales_raw)
        print(f"[3DGS] Scales treated as log-stddev (neg_frac={frac_neg_scale:.3f})")
    else:
        scales = scales_raw
        print(f"[3DGS] Scales treated as linear (neg_frac={frac_neg_scale:.3f})")

    # Heuristic for opacities (logits -> sigmoid if outside [0,1]) - НОВОЕ
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
        opacities = opacities[idx]  # НОВОЕ: downsample opacity тоже
        colors = colors[idx]
        
        print(f"[3DGS] Downsampled {N} -> {max_points} points")

    # --- Convert to torch tensors ---
    means_tensor = torch.tensor(means, dtype=torch.float32).to(device)    # [N, 3]
    quats_tensor = torch.tensor(quats, dtype=torch.float32).to(device)    # [N, 4] 
    scales_tensor = torch.tensor(scales, dtype=torch.float32).to(device)  # [N, 3]
    colors_tensor = torch.tensor(colors, dtype=torch.float32).to(device)  # [N, 3]
    opacities_tensor = torch.tensor(opacities, dtype=torch.float32).to(device)  # НОВОЕ: реальные opacity

    # Create GaussianData
    gaussian_data = GaussianData(
        means=means_tensor,
        quats=quats_tensor,
        scales=scales_tensor,
        colors=colors_tensor,
    )

    # Добавляем реальные opacities как атрибут - НОВОЕ
    gaussian_data.real_opacities = opacities_tensor

    # Validate shapes
    if not gaussian_data.validate_shapes():
        raise ValueError("Inconsistent tensor shapes after 3DGS loading")

    print(f"[3DGS] Successfully loaded {len(gaussian_data.means)} points with real opacities")
    return gaussian_data

def smooth_path(poses: List[dict], smoothness: float = 0.5) -> List[dict]:
    """
    Сглаживает путь с плавными поворотами камеры
    """
    if len(poses) <= 2:
        return poses
    
    # Извлекаем позиции и направления
    camera_positions = np.array([p['camera_position'] for p in poses])
    look_at_points = np.array([p['look_at'] for p in poses])
    
    # Вычисляем направления взгляда (векторы)
    look_directions = look_at_points - camera_positions
    
    # Нормализуем направления
    look_norms = np.linalg.norm(look_directions, axis=1, keepdims=True)
    look_norms[look_norms == 0] = 1.0  # избегаем деления на ноль
    look_directions_norm = look_directions / look_norms
    
    # Вычисляем кумулятивное расстояние
    distances = np.sqrt(np.sum(np.diff(camera_positions, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
    total_length = cumulative_dist[-1]
    
    # Создаем равномерные параметры
    original_params = cumulative_dist / total_length if total_length > 0 else np.linspace(0, 1, len(poses))
    num_output_points = len(poses)
    uniform_params = np.linspace(0, 1, num_output_points)
    
    # Интерполяция позиций камеры (линейная + сглаживание)
    smooth_camera_positions = []
    smooth_look_at_points = []
    
    for t in uniform_params:
        # Находим индексы для интерполяции
        idx = np.searchsorted(original_params, t) - 1
        idx = max(0, min(idx, len(original_params) - 2))
        
        # Локальный параметр между опорными точками
        t_local = (t - original_params[idx]) / (original_params[idx + 1] - original_params[idx])
        t_local = np.clip(t_local, 0, 1)
        
        # Интерполяция позиции камеры (кубическое сглаживание)
        t_smooth = t_local * t_local * (3 - 2 * t_local)  # smoothstep функция
        
        camera_pos = (1 - t_smooth) * camera_positions[idx] + t_smooth * camera_positions[idx + 1]
        smooth_camera_positions.append(camera_pos)
        
        # СФЕРИЧЕСКАЯ ИНТЕРПОЛЯЦИЯ направлений (ручная реализация SLERP)
        dir1 = look_directions_norm[idx]
        dir2 = look_directions_norm[idx + 1]
        
        # Косинус угла между направлениями
        dot_product = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
        angle = math.acos(dot_product)
        
        if angle < 1e-6:
            # Направления почти одинаковые
            smooth_dir = dir1
        else:
            # Ручная реализация SLERP
            sin_angle = math.sin(angle)
            weight1 = math.sin((1 - t_smooth) * angle) / sin_angle
            weight2 = math.sin(t_smooth * angle) / sin_angle
            smooth_dir = weight1 * dir1 + weight2 * dir2
        
        # Применяем сглаженное направление
        look_ahead_distance = 3.0
        smooth_look_at = camera_pos + smooth_dir * look_ahead_distance
        smooth_look_at_points.append(smooth_look_at)
    
    # Собираем обратно в словари
    smooth_poses = []
    for i in range(num_output_points):
        smooth_poses.append({
            'camera_position': smooth_camera_positions[i].tolist(),
            'look_at': smooth_look_at_points[i].tolist()
        })
    
    return smooth_poses

def generate_straight_path(start_pos: List[float], end_pos: List[float], num_points: int = 40, next_point: Optional[List[float]] = None) -> List[dict]:
    """
    Генерирует прямой путь с плавными поворотами
    next_point: следующая точка после end_pos для плавного предвосхищения поворота
    """
    poses = []
    
    start_arr = np.array(start_pos)
    end_arr = np.array(end_pos)
    
    # Вычисляем направление движения
    movement_vector = end_arr - start_arr
    movement_length = np.linalg.norm(movement_vector)
    
    if movement_length > 0:
        movement_direction = movement_vector / movement_length
    else:
        movement_direction = np.array([1.0, 0.0, 0.0])  # fallback
    
    # Если есть следующая точка, вычисляем направление к ней для плавного поворота
    turn_direction = None
    if next_point is not None:
        next_arr = np.array(next_point)
        turn_vector = next_arr - end_arr
        turn_length = np.linalg.norm(turn_vector)
        if turn_length > 0:
            turn_direction = turn_vector / turn_length
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        
        # Плавная интерполяция позиции
        t_smooth = t * t * (3 - 2 * t)  # smoothstep
        
        camera_pos = [
            start_pos[0] + t_smooth * (end_pos[0] - start_pos[0]),
            start_pos[1] + t_smooth * (end_pos[1] - start_pos[1]),
            start_pos[2] + t_smooth * (end_pos[2] - start_pos[2])
        ]
        
        # ПЛАВНОЕ ИЗМЕНЕНИЕ НАПРАВЛЕНИЯ ВЗГЛЯДА
        look_ahead_distance = 3.0
        
        if turn_direction is not None:
            # Плавно переходим от движения к повороту
            turn_weight = max(0, (t - 0.5) * 2)  # Во второй половине пути начинаем поворот
            turn_weight = turn_weight * turn_weight  # Квадратичное сглаживание
            
            # Интерполируем между текущим направлением и направлением поворота
            blended_direction = (1 - turn_weight) * movement_direction + turn_weight * turn_direction
            blended_direction = blended_direction / np.linalg.norm(blended_direction)
            
            look_at = [
                camera_pos[0] + blended_direction[0] * look_ahead_distance,
                camera_pos[1] + 1.0,
                camera_pos[2] + blended_direction[2] * look_ahead_distance
            ]
        else:
            # Без поворота - смотрим строго по направлению движения
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
    Генерирует путь через несколько точек с ПЛАВНЫМИ переходами между сегментами
    """
    if len(waypoints) < 2:
        raise ValueError("Нужно как минимум 2 точки-ориентира")
    
    all_poses = []
    
    # Генерируем сегменты с информацией о поворотах
    for i in range(len(waypoints) - 1):
        start_point = waypoints[i]
        end_point = waypoints[i + 1]
        
        print(f"Создаем сегмент {i+1}/{len(waypoints)-1}: {start_point} -> {end_point}")
        
        # Определяем следующую точку для плавного поворота (если есть)
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
            # Пропускаем первую точку чтобы избежать дублирования
            all_poses.extend(segment_poses[1:])
    
    print(f"Сгенерирован путь через {len(waypoints)} точек, всего {len(all_poses)} позиций камеры")
    return all_poses

def generate_elliptical_path(
    ellipse_center_xz: List[float] = [0.0, 0.0],  # Центр эллипса в плоскости XZ
    radius_x: float = 2.0,                         # Радиус по оси X (ширина)
    radius_z: float = 1.0,                         # Радиус по оси Z (высота)
    camera_height: float = 0.4,                    # Высота камеры (постоянная)
    num_points: int = 60,
    start_angle: float = 0.0,
    end_angle: float = 2 * math.pi
) -> List[dict]:
    """
    Генерирует эллиптический путь параллельно плоскости XZ, камера всегда смотрит ВНУТРЬ
    """
    poses = []
    
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        angle = start_angle + t * (end_angle - start_angle)
        
        # Позиция камеры на эллипсе в плоскости XZ
        camera_x = ellipse_center_xz[0] + radius_x * math.cos(angle)
        camera_z = ellipse_center_xz[1] + radius_z * math.sin(angle)
        camera_y = camera_height  # Постоянная высота
        
        # Для правильного направления взгляда вычисляем нормаль к эллипсу
        # Производные параметрического уравнения эллипса
        dx_dt = -radius_x * math.sin(angle)  # производная по x
        dz_dt = radius_z * math.cos(angle)   # производная по z
        
        # Нормаль к эллипсу (перпендикулярно касательной)
        normal_x = -dz_dt
        normal_z = dx_dt
        
        # Нормализуем нормаль
        normal_length = math.sqrt(normal_x**2 + normal_z**2)
        if normal_length > 0:
            normal_x /= normal_length
            normal_z /= normal_length
        
        # Look-at точка - смещаемся по нормали внутрь эллипса
        look_ahead_distance = 3  # расстояние для look-at точки
        look_at_x = camera_x + normal_x * look_ahead_distance
        look_at_z = camera_z + normal_z * look_ahead_distance
        look_at_y = camera_height + 1 # ТА ЖЕ ВЫСОТА!
        
        poses.append({
            'camera_position': [camera_x, camera_y, camera_z],
            'look_at': [look_at_x, look_at_y, look_at_z]
        })
    
    print(f"Эллиптический путь: центр={ellipse_center_xz}, радиусы X/Z={radius_x}/{radius_z}, высота={camera_height}")
    return poses

if __name__ == "__main__":
    try:
        # 1. Загрузка данных
        print("Загрузка PLY файла...")
        gaussian_data = load_gaussians_from_3dgs_ply("inputs/outdoor-drone_open.ply")
        
        print(f"Загружено {len(gaussian_data.means)} гауссов")
        
        # 2. Генерируем круговой путь
        print("Генерация кругового пути...")
        # Эллипс с шириной 2 и высотой 1
        elliptical_poses = generate_elliptical_path(
            ellipse_center_xz=[0.0, 0.0],
            radius_x=1.8,    # ширина по X
            radius_z=0.8,    # высота по Z  
            camera_height=0,
            num_points=400
        )
        
        print(f"Сгенерировано {len(elliptical_poses)} точек кругового пути")
        
        # 3. Сглаживаем путь (для круга можно уменьшить сглаживание или пропустить)
        print("Сглаживание пути...")
        smooth_poses = smooth_path(elliptical_poses, smoothness=0.3)  # Меньше сглаживания для круга
        
        print(f"После сглаживания: {len(smooth_poses)} точек")
        
        # 4. Создание видео
        print("Создание видео...")
        create_video_from_poses(
            gaussian_data=gaussian_data,
            poses=smooth_poses,
            output_path="outputs/circular_walkthrough.mp4",
            image_size=(1280, 720),
            fov_degrees=80.0,  # Больше FOV для лучшего обзора при круговом движении
            fps=30,  # Более плавное видео для кругового движения
            enable_object_detection=True 
        )
        
        print("Готово! Видео сохранено как 'outputs/circular_walkthrough.mp4'")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
    try:
        # 1. Загрузка данных
        print("Загрузка PLY файла...")
        gaussian_data = load_gaussians_from_3dgs_ply("inputs/Museume_open.ply")
        
        print(f"Загружено {len(gaussian_data.means)} гауссов")
        
        # 2. Задаем точки пути
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
                    [-10, -3, -1]]
        
        print(f"Точки пути: {len(waypoints)} точек")
        
        # 3. Генерируем многоточечный путь
        print("Генерация многоточечного пути...")
        multi_poses = generate_multi_point_path(
            waypoints=waypoints,
            points_per_segment=200
        )
        
        print(f"Сгенерировано {len(multi_poses)} точек пути")
        
        # 4. Сглаживаем путь для плавности
        print("Сглаживание пути...")
        smooth_poses = smooth_path(multi_poses, smoothness=0.7)
        
        print(f"После сглаживания: {len(smooth_poses)} точек")
        
        # 5. Создание видео
        print("Создание видео...")
        create_video_from_poses(
            gaussian_data=gaussian_data,
            poses=smooth_poses,
            output_path="outputs/multi_point_walkthrough.mp4",
            image_size=(1280, 720),
            fov_degrees=120.0,
            fps=30, 
            enable_object_detection=True 
        )
        
        print("Готово! Видео сохранено как 'outputs/multi_point_walkthrough.mp4'")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()