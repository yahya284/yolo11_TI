import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Position:
    TOP_LEFT = 0
    TOP_CENTER = 1
    TOP_RIGHT = 2
    CENTER_LEFT = 3
    CENTER = 4
    CENTER_RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM_CENTER = 7
    BOTTOM_RIGHT = 8

class HeatMapAnnotator:
    def __init__(
        self,
        position: Position = Position.BOTTOM_CENTER,
        opacity: float = 0.5,
        radius: int = 25,
        kernel_size: int = 25,
        top_hue: int = 0,    # Red
        low_hue: int = 125   # Blue
    ):
        self._position = position
        self._opacity = opacity
        self._radius = radius
        self._kernel_size = kernel_size
        self._top_hue = top_hue
        self._low_hue = low_hue
        self._heat_matrix = None
        
    def _initialize_heat_matrix(self, frame_resolution: Tuple[int, int]) -> None:
        if self._heat_matrix is None:
            self._heat_matrix = np.zeros(frame_resolution, dtype=np.float32)
            
    def _apply_gaussian_blur(self) -> None:
        kernel_size = self._kernel_size + (1 - self._kernel_size % 2)
        self._heat_matrix = cv2.GaussianBlur(
            self._heat_matrix, 
            (kernel_size, kernel_size), 
            0
        )
        
    def _create_overlay(self, frame_resolution: Tuple[int, int]) -> np.ndarray:
        normalized = cv2.normalize(self._heat_matrix, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # Create HSV heatmap
        heatmap = np.zeros((frame_resolution[0], frame_resolution[1], 3), dtype=np.uint8)
        heatmap[..., 0] = np.clip(
            self._low_hue + (self._top_hue - self._low_hue) * 
            (255 - normalized) / 255, 
            0, 
            179
        )
        heatmap[..., 1] = 255
        heatmap[..., 2] = normalized
        
        return cv2.cvtColor(heatmap, cv2.COLOR_HSV2BGR)
        
    def annotate(
        self,
        scene: np.ndarray,
        detections,
        filter_class_id: Optional[int] = None
    ) -> np.ndarray:
        self._initialize_heat_matrix((scene.shape[0], scene.shape[1]))
        
        # Appliquer un facteur de décroissance pour effacer les traces anciennes
        self._heat_matrix *= 1  # Ajustez ce facteur entre 0 et 1 pour un effacement plus ou moins rapide.
        
        # Mettre à jour la matrice de chaleur avec les nouvelles détections
        for xyxy, confidence, class_id, _ in detections:
            if filter_class_id is not None and class_id != filter_class_id:
                continue
                
            x_center = int((xyxy[0] + xyxy[2]) / 2)
            y_center = int((xyxy[1] + xyxy[3]) / 2)
            
            cv2.circle(
                self._heat_matrix,
                (x_center, y_center),
                self._radius,
                confidence,
                -1
            )
        
        self._apply_gaussian_blur()
        heatmap = self._create_overlay((scene.shape[0], scene.shape[1]))
        
        # Mélanger la carte thermique avec la scène d'origine
        return cv2.addWeighted(scene, 1, heatmap, self._opacity, 0)
