"""
图像处理辅助模块
提供基础的图像处理功能，支持车牌图像的预处理
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
import logging

class ImageProcessor:
    """
    图像处理器
    提供基础的图像处理功能
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int], 
                    maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 输入图像
            target_size: 目标尺寸 (width, height)
            maintain_aspect_ratio: 是否保持宽高比
            
        Returns:
            调整大小后的图像
        """
        if maintain_aspect_ratio:
            return self._resize_with_aspect_ratio(image, target_size)
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    
    def _resize_with_aspect_ratio(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """保持宽高比的图像缩放"""
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        
        # 计算新的尺寸
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # 创建目标尺寸的图像并将缩放后的图像居中放置
        if len(image.shape) == 3:
            result = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            result = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # 计算偏移量
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # 放置图像
        if len(image.shape) == 3:
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        else:
            result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return result
    
    def rotate_image(self, image: np.ndarray, angle: float, scale: float = 1.0) -> np.ndarray:
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
            scale: 缩放因子
            
        Returns:
            旋转后的图像
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 获取旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 计算新的边界框
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # 调整旋转中心
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # 执行旋转
        rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h))
        
        return rotated
    
    def correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        纠正图像倾斜
        
        Args:
            image: 输入图像
            
        Returns:
            纠正后的图像和倾斜角度
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None:
            return image, 0.0
        
        # 计算平均角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # 只考虑接近水平的线
            if abs(angle - 90) < 45:
                angles.append(angle - 90)
            elif abs(angle - 0) < 45:
                angles.append(angle)
        
        if not angles:
            return image, 0.0
        
        # 使用中位数作为倾斜角度
        skew_angle = np.median(angles)
        
        # 纠正倾斜
        corrected = self.rotate_image(image, -skew_angle)
        
        return corrected, skew_angle
    
    def enhance_local_contrast(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        增强局部对比度
        
        Args:
            image: 输入图像
            kernel_size: 内核大小
            
        Returns:
            增强后的图像
        """
        if len(image.shape) == 3:
            # 对于彩色图像，在LAB空间中处理亮度通道
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # 应用局部对比度增强
            enhanced_l = self._apply_local_contrast(l_channel, kernel_size)
            
            # 合并通道
            lab[:, :, 0] = enhanced_l
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像直接处理
            enhanced = self._apply_local_contrast(image, kernel_size)
        
        return enhanced
    
    def _apply_local_contrast(self, channel: np.ndarray, kernel_size: int) -> np.ndarray:
        """应用局部对比度增强"""
        # 计算局部均值
        mean = cv2.blur(channel.astype(np.float32), (kernel_size, kernel_size))
        
        # 计算局部方差
        sqr_mean = cv2.blur((channel.astype(np.float32)) ** 2, (kernel_size, kernel_size))
        variance = sqr_mean - mean ** 2
        std = np.sqrt(np.maximum(variance, 0))
        
        # 避免除零
        std = np.maximum(std, 1.0)
        
        # 应用局部对比度增强
        enhanced = (channel.astype(np.float32) - mean) / std * 50 + 128
        
        # 限制在有效范围内
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        去除阴影
        
        Args:
            image: 输入图像
            
        Returns:
            去除阴影后的图像
        """
        # 转换到LAB颜色空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # 使用形态学操作创建背景模型
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(l_channel, cv2.MORPH_OPEN, kernel)
        
        # 背景减法
        diff = cv2.subtract(l_channel, background)
        
        # 标准化
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        # 替换L通道
        lab[:, :, 0] = normalized
        
        # 转换回BGR
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def enhance_edges(self, image: np.ndarray, method: str = 'laplacian') -> np.ndarray:
        """
        边缘增强
        
        Args:
            image: 输入图像
            method: 边缘检测方法 ('laplacian', 'sobel', 'scharr')
            
        Returns:
            边缘增强后的图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'laplacian':
            edges = cv2.Laplacian(gray, cv2.CV_64F)
        elif method == 'sobel':
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobel_x**2 + sobel_y**2)
        elif method == 'scharr':
            scharr_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            scharr_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            edges = np.sqrt(scharr_x**2 + scharr_y**2)
        else:
            raise ValueError(f"未支持的边缘检测方法: {method}")
        
        # 标准化边缘图像
        edges = np.absolute(edges)
        edges = np.uint8(np.clip(edges, 0, 255))
        
        # 将边缘添加到原图
        if len(image.shape) == 3:
            enhanced = image.copy()
            for i in range(3):
                enhanced[:, :, i] = cv2.addWeighted(image[:, :, i], 0.8, edges, 0.2, 0)
        else:
            enhanced = cv2.addWeighted(image, 0.8, edges, 0.2, 0)
        
        return enhanced
    
    def histogram_equalization(self, image: np.ndarray, method: str = 'global') -> np.ndarray:
        """
        直方图均衡化
        
        Args:
            image: 输入图像
            method: 均衡化方法 ('global', 'adaptive')
            
        Returns:
            均衡化后的图像
        """
        if len(image.shape) == 3:
            # 彩色图像，在YUV空间中处理
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            
            if method == 'global':
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            elif method == 'adaptive':
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
            
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # 灰度图像
            if method == 'global':
                enhanced = cv2.equalizeHist(image)
            elif method == 'adaptive':
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            else:
                raise ValueError(f"未支持的均衡化方法: {method}")
        
        return enhanced
    
    def gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        伽马校正
        
        Args:
            image: 输入图像
            gamma: 伽马值
            
        Returns:
            校正后的图像
        """
        # 构建查找表
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        
        # 应用伽马校正
        corrected = cv2.LUT(image, table)
        
        return corrected 