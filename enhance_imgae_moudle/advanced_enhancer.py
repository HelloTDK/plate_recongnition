"""
高级车牌图像增强模块
包含专门针对车牌识别的高级优化技术
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict
import logging
from scipy import ndimage
from skimage import restoration, morphology

class AdvancedPlateEnhancer:
    """
    高级车牌增强器
    包含专门的车牌识别优化算法
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        专门为OCR识别优化的图像增强
        
        Args:
            image: 输入图像
            
        Returns:
            OCR优化后的图像
        """
        enhanced = image.copy()
        
        # 1. 字符分离增强
        enhanced = self._enhance_character_separation(enhanced)
        
        # 2. 字符清晰化
        enhanced = self._enhance_character_clarity(enhanced)
        
        # 3. 背景统一化
        enhanced = self._normalize_background(enhanced)
        
        # 4. 字符粗细标准化
        enhanced = self._normalize_character_thickness(enhanced)
        
        return enhanced
    
    def _enhance_character_separation(self, image: np.ndarray) -> np.ndarray:
        """增强字符之间的分离度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用形态学操作增强字符分离
        # 垂直线检测核
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        
        # 开运算去除连接字符的细线
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        
        # 增强字符边界
        edges = cv2.Canny(opened, 50, 150)
        
        # 膨胀边缘以增强分离
        kernel = np.ones((2, 1), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # 将边缘信息融合回原图
        result = cv2.subtract(gray, dilated_edges)
        
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def _enhance_character_clarity(self, image: np.ndarray) -> np.ndarray:
        """增强字符清晰度"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用非锐化遮罩增强细节
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
        
        # 限制过度增强
        enhanced = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def _normalize_background(self, image: np.ndarray) -> np.ndarray:
        """标准化背景，突出前景字符"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用大核的开运算估计背景
        background_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, background_kernel)
        
        # 背景减法
        foreground = cv2.subtract(gray, background)
        
        # 增强前景对比度
        normalized = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)
        
        if len(image.shape) == 3:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        
        return normalized
    
    def _normalize_character_thickness(self, image: np.ndarray) -> np.ndarray:
        """标准化字符粗细"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 距离变换
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # 根据距离变换调整字符粗细
        # 较粗的字符会被细化，较细的会被加粗
        mean_thickness = np.mean(dist[dist > 0])
        
        if mean_thickness > 3:
            # 字符太粗，使用腐蚀
            kernel = np.ones((2, 2), np.uint8)
            result = cv2.erode(binary, kernel, iterations=1)
        elif mean_thickness < 1.5:
            # 字符太细，使用膨胀
            kernel = np.ones((2, 2), np.uint8)
            result = cv2.dilate(binary, kernel, iterations=1)
        else:
            result = binary
        
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def remove_motion_blur(self, image: np.ndarray, blur_length: int = 15, 
                          blur_angle: float = 0) -> np.ndarray:
        """
        去除运动模糊
        
        Args:
            image: 输入图像
            blur_length: 模糊长度
            blur_angle: 模糊角度
            
        Returns:
            去模糊后的图像
        """
        if len(image.shape) == 3:
            # 对每个通道分别处理
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = self._deblur_channel(image[:, :, i], blur_length, blur_angle)
            return result
        else:
            return self._deblur_channel(image, blur_length, blur_angle)
    
    def _deblur_channel(self, channel: np.ndarray, blur_length: int, blur_angle: float) -> np.ndarray:
        """对单通道进行去模糊"""
        # 创建运动模糊核
        kernel = self._create_motion_blur_kernel(blur_length, blur_angle)
        
        # 使用维纳滤波进行反卷积
        try:
            # 转换为浮点数
            channel_float = channel.astype(np.float64) / 255.0
            
            # 维纳滤波
            noise_power = 0.01  # 噪声功率估计
            deblurred = restoration.wiener(channel_float, kernel, noise_power)
            
            # 转换回uint8
            deblurred = np.clip(deblurred * 255, 0, 255).astype(np.uint8)
            
            return deblurred
        except Exception as e:
            self.logger.warning(f"维纳滤波失败，使用替代方法: {e}")
            # 使用简单的锐化作为备选
            return self._simple_sharpen(channel)
    
    def _create_motion_blur_kernel(self, length: int, angle: float) -> np.ndarray:
        """创建运动模糊核"""
        kernel = np.zeros((length, length))
        
        # 计算中心点
        center = length // 2
        
        # 根据角度创建线性核
        cos_angle = np.cos(np.radians(angle))
        sin_angle = np.sin(np.radians(angle))
        
        for i in range(length):
            offset = i - center
            x = int(center + offset * cos_angle)
            y = int(center + offset * sin_angle)
            
            if 0 <= x < length and 0 <= y < length:
                kernel[y, x] = 1
        
        # 归一化
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _simple_sharpen(self, channel: np.ndarray) -> np.ndarray:
        """简单锐化"""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(channel, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    def enhance_low_resolution(self, image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
        """
        低分辨率图像增强（超分辨率）
        
        Args:
            image: 输入图像
            scale_factor: 放大倍数
            
        Returns:
            增强后的高分辨率图像
        """
        # 使用双三次插值进行初始放大
        h, w = image.shape[:2]
        new_size = (w * scale_factor, h * scale_factor)
        upscaled = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
        
        # 应用边缘保持滤波
        enhanced = self._edge_preserving_filter(upscaled)
        
        # 锐化细节
        sharpened = self._selective_sharpening(enhanced)
        
        return sharpened
    
    def _edge_preserving_filter(self, image: np.ndarray) -> np.ndarray:
        """边缘保持滤波"""
        # 使用双边滤波保持边缘的同时平滑噪声
        filtered = cv2.bilateralFilter(image, 9, 80, 80)
        return filtered
    
    def _selective_sharpening(self, image: np.ndarray) -> np.ndarray:
        """选择性锐化，主要锐化边缘区域"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测边缘
        edges = cv2.Canny(gray, 50, 150)
        
        # 创建边缘掩码
        edge_mask = edges / 255.0
        
        # 锐化核
        sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        
        if len(image.shape) == 3:
            sharpened = np.zeros_like(image)
            for i in range(3):
                channel_sharpened = cv2.filter2D(image[:, :, i], -1, sharpening_kernel)
                # 只在边缘区域应用锐化
                sharpened[:, :, i] = (image[:, :, i] * (1 - edge_mask) + 
                                    channel_sharpened * edge_mask).astype(np.uint8)
        else:
            channel_sharpened = cv2.filter2D(image, -1, sharpening_kernel)
            sharpened = (image * (1 - edge_mask) + channel_sharpened * edge_mask).astype(np.uint8)
        
        return sharpened
    
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        光照不均匀校正
        
        Args:
            image: 输入图像
            
        Returns:
            光照校正后的图像
        """
        if len(image.shape) == 3:
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
        else:
            l_channel = image.copy()
        
        # 估计背景光照
        # 使用形态学操作估计背景
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        background = cv2.morphologyEx(l_channel, cv2.MORPH_OPEN, kernel)
        
        # 高斯模糊平滑背景
        background = cv2.GaussianBlur(background, (51, 51), 0)
        
        # 计算校正图像
        corrected = cv2.divide(l_channel.astype(np.float32), 
                              background.astype(np.float32), scale=255)
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        if len(image.shape) == 3:
            # 替换L通道
            lab[:, :, 0] = corrected
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = corrected
        
        return result
    
    def enhance_text_regions(self, image: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        检测并增强文本区域
        
        Args:
            image: 输入图像
            
        Returns:
            增强后的图像和文本区域边界框列表
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 检测文本区域
        text_regions = self._detect_text_regions(gray)
        
        # 对每个文本区域进行专门增强
        enhanced = image.copy()
        for x, y, w, h in text_regions:
            roi = enhanced[y:y+h, x:x+w]
            enhanced_roi = self._enhance_text_roi(roi)
            enhanced[y:y+h, x:x+w] = enhanced_roi
        
        return enhanced, text_regions
    
    def _detect_text_regions(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测文本区域"""
        # 使用MSER检测文本区域
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # 过滤和合并区域
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # 过滤条件：适当的宽高比和大小
            aspect_ratio = w / h
            area = w * h
            
            if (0.1 < aspect_ratio < 10 and 100 < area < 5000):
                text_regions.append((x, y, w, h))
        
        # 合并重叠的区域
        merged_regions = self._merge_overlapping_regions(text_regions)
        
        return merged_regions
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """合并重叠的区域"""
        if not regions:
            return []
        
        # 简单的合并策略：如果两个区域重叠超过阈值，则合并
        merged = []
        used = set()
        
        for i, region1 in enumerate(regions):
            if i in used:
                continue
            
            x1, y1, w1, h1 = region1
            merged_region = [x1, y1, x1 + w1, y1 + h1]  # [xmin, ymin, xmax, ymax]
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue
                
                x2, y2, w2, h2 = region2
                
                # 计算重叠面积
                overlap_x = max(0, min(merged_region[2], x2 + w2) - max(merged_region[0], x2))
                overlap_y = max(0, min(merged_region[3], y2 + h2) - max(merged_region[1], y2))
                overlap_area = overlap_x * overlap_y
                
                area1 = (merged_region[2] - merged_region[0]) * (merged_region[3] - merged_region[1])
                area2 = w2 * h2
                union_area = area1 + area2 - overlap_area
                
                if overlap_area / union_area > 0.3:  # 重叠阈值
                    # 合并区域
                    merged_region[0] = min(merged_region[0], x2)
                    merged_region[1] = min(merged_region[1], y2)
                    merged_region[2] = max(merged_region[2], x2 + w2)
                    merged_region[3] = max(merged_region[3], y2 + h2)
                    used.add(j)
            
            # 转换回 (x, y, w, h) 格式
            x, y = merged_region[0], merged_region[1]
            w, h = merged_region[2] - merged_region[0], merged_region[3] - merged_region[1]
            merged.append((x, y, w, h))
            used.add(i)
        
        return merged
    
    def _enhance_text_roi(self, roi: np.ndarray) -> np.ndarray:
        """增强文本感兴趣区域"""
        # 对文本区域应用专门的增强
        enhanced = self.enhance_for_ocr(roi)
        
        # 额外的文本特定增强
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced.copy()
        
        # 自适应阈值化
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 形态学清理
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        if len(enhanced.shape) == 3:
            cleaned = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return cleaned
    
    def adaptive_enhancement_pipeline(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        自适应增强流水线，根据图像质量自动选择合适的增强方法
        
        Args:
            image: 输入图像
            
        Returns:
            包含不同增强结果的字典
        """
        results = {'original': image.copy()}
        
        # 分析图像质量
        quality_metrics = self._analyze_image_quality(image)
        
        # 基于质量指标选择增强策略
        if quality_metrics['blur_level'] > 0.3:
            results['deblurred'] = self.remove_motion_blur(image)
        
        if quality_metrics['resolution_quality'] < 0.5:
            results['super_resolution'] = self.enhance_low_resolution(image)
        
        if quality_metrics['illumination_uniformity'] < 0.7:
            results['illumination_corrected'] = self.correct_illumination(image)
        
        # OCR优化
        results['ocr_optimized'] = self.enhance_for_ocr(image)
        
        # 文本区域增强
        enhanced_text, text_regions = self.enhance_text_regions(image)
        results['text_enhanced'] = enhanced_text
        results['text_regions'] = text_regions
        
        return results
    
    def _analyze_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        分析图像质量指标
        
        Args:
            image: 输入图像
            
        Returns:
            质量指标字典
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 模糊程度检测（使用拉普拉斯方差）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_level = 1.0 - min(laplacian_var / 1000.0, 1.0)  # 归一化到0-1
        
        # 分辨率质量（基于边缘密度）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        resolution_quality = min(edge_density * 10, 1.0)  # 归一化到0-1
        
        # 光照均匀性（基于标准差）
        illumination_std = np.std(gray)
        illumination_uniformity = 1.0 - min(illumination_std / 100.0, 1.0)  # 归一化到0-1
        
        # 噪声级别估计
        noise_level = self._estimate_noise_level(gray)
        
        return {
            'blur_level': blur_level,
            'resolution_quality': resolution_quality,
            'illumination_uniformity': illumination_uniformity,
            'noise_level': noise_level
        }
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """估计噪声级别"""
        # 使用高通滤波估计噪声
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4.0
        noise_response = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        noise_level = np.std(noise_response) / 255.0  # 归一化到0-1
        
        return min(noise_level, 1.0) 