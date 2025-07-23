"""
车牌图像增强主模块
集成多种图像增强技术，专门针对车牌识别优化
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
from .image_processor import ImageProcessor
from .noise_reducer import NoiseReducer

class PlateEnhancer:
    """
    车牌图像增强器
    提供多种增强策略来提高车牌识别准确性
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        初始化车牌增强器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or self._get_default_config()
        self.image_processor = ImageProcessor()
        self.noise_reducer = NoiseReducer()
        self.logger = logging.getLogger(__name__)
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 64,  # 目标高度
            'clahe_clip_limit': 3.0,
            'gaussian_kernel_size': 3,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
        }
    
    def enhance_plate_image(self, image: np.ndarray, enhancement_level: str = 'medium') -> np.ndarray:
        """
        增强车牌图像
        
        Args:
            image: 输入图像 (BGR格式)
            enhancement_level: 增强级别 ('light', 'medium', 'strong')
            
        Returns:
            增强后的图像
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")
        
        enhanced_image = image.copy()
        
        try:
            # 1. 尺寸标准化
            enhanced_image = self._normalize_size(enhanced_image)
            
            # 2. 亮度和对比度标准化
            if self.config['enable_brightness_normalization']:
                enhanced_image = self._normalize_brightness_contrast(enhanced_image)
            
            # 3. 噪声减少
            if self.config['enable_noise_reduction']:
                enhanced_image = self.noise_reducer.reduce_noise(enhanced_image, method='bilateral')
            
            # 4. 对比度增强
            if self.config['enable_contrast_enhancement']:
                enhanced_image = self._enhance_contrast(enhanced_image, enhancement_level)
            
            # 5. 锐化处理
            if self.config['enable_sharpening']:
                enhanced_image = self._sharpen_image(enhanced_image, enhancement_level)
            
            # 6. 自适应阈值化
            if self.config['enable_adaptive_thresholding']:
                enhanced_image = self._apply_adaptive_threshold(enhanced_image)
            
            # 7. 透视校正
            if self.config['enable_perspective_correction']:
                enhanced_image = self._correct_perspective(enhanced_image)
            
            return enhanced_image
            
        except Exception as e:
            self.logger.error(f"图像增强过程中发生错误: {e}")
            return image
    
    def _normalize_size(self, image: np.ndarray) -> np.ndarray:
        """标准化图像尺寸"""
        height, width = image.shape[:2]
        target_height = self.config['target_height']
        
        if height != target_height:
            aspect_ratio = width / height
            target_width = int(target_height * aspect_ratio)
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        return image
    
    def _normalize_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """亮度和对比度标准化"""
        # 转换为灰度图进行分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算当前亮度和对比度
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)
        
        # 目标亮度和对比度
        target_brightness = 128
        target_contrast = 50
        
        # 调整亮度
        brightness_factor = target_brightness - mean_brightness
        
        # 调整对比度
        if std_contrast > 0:
            contrast_factor = target_contrast / std_contrast
        else:
            contrast_factor = 1.0
        
        # 应用调整
        enhanced = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
        
        return enhanced
    
    def _enhance_contrast(self, image: np.ndarray, level: str) -> np.ndarray:
        """增强对比度"""
        # 使用CLAHE (对比度限制自适应直方图均衡化)
        clip_limit = {
            'light': 2.0,
            'medium': 3.0,
            'strong': 4.0
        }.get(level, 3.0)
        
        if len(image.shape) == 3:
            # 彩色图像，在LAB颜色空间中处理
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def _sharpen_image(self, image: np.ndarray, level: str) -> np.ndarray:
        """图像锐化"""
        # 根据增强级别选择锐化核
        kernels = {
            'light': np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]),
            'medium': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
            'strong': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        }
        
        kernel = kernels.get(level, kernels['medium'])
        enhanced = cv2.filter2D(image, -1, kernel)
        
        return enhanced
    
    def _apply_adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """应用自适应阈值化"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 应用高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值化
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学操作以清理图像
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # 如果原图是彩色的，转换回彩色格式
        if len(image.shape) == 3:
            thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        
        return thresh
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """透视校正"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # 找到最大的矩形轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 近似轮廓为多边形
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # 如果找到了四边形，进行透视变换
        if len(approx) == 4:
            # 排序角点
            pts = self._order_points(approx.reshape(4, 2))
            
            # 计算目标尺寸
            width = max(
                np.linalg.norm(pts[1] - pts[0]),
                np.linalg.norm(pts[3] - pts[2])
            )
            height = max(
                np.linalg.norm(pts[3] - pts[0]),
                np.linalg.norm(pts[2] - pts[1])
            )
            
            # 目标点
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(pts, dst)
            
            # 应用透视变换
            corrected = cv2.warpPerspective(image, M, (int(width), int(height)))
            return corrected
        
        return image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """按顺序排列四个角点：左上，右上，右下，左下"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 左上角点的和最小，右下角点的和最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # 右上角点的差值最小，左下角点的差值最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def batch_enhance(self, images: List[np.ndarray], enhancement_level: str = 'medium') -> List[np.ndarray]:
        """
        批量增强图像
        
        Args:
            images: 图像列表
            enhancement_level: 增强级别
            
        Returns:
            增强后的图像列表
        """
        enhanced_images = []
        
        for i, image in enumerate(images):
            try:
                enhanced = self.enhance_plate_image(image, enhancement_level)
                enhanced_images.append(enhanced)
                self.logger.info(f"成功增强第 {i+1} 张图像")
            except Exception as e:
                self.logger.error(f"增强第 {i+1} 张图像时发生错误: {e}")
                enhanced_images.append(image)  # 返回原图
        
        return enhanced_images
    
    def get_enhancement_preview(self, image: np.ndarray) -> dict:
        """
        获取不同增强级别的预览
        
        Args:
            image: 输入图像
            
        Returns:
            包含不同级别增强结果的字典
        """
        preview = {
            'original': image.copy(),
            'light': self.enhance_plate_image(image, 'light'),
            'medium': self.enhance_plate_image(image, 'medium'),
            'strong': self.enhance_plate_image(image, 'strong')
        }
        
        return preview 