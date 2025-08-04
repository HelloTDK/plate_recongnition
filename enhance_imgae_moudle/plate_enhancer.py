"""
车牌图像增强主模块
集成多种图像增强技术，专门针对车牌识别优化
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
import logging
import os
import glob
from pathlib import Path
from image_processor import ImageProcessor
from noise_reducer import NoiseReducer

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
            'enable_adaptive_thresholding': False,  # 关闭二值化，保持原始颜色
            'enable_contrast_enhancement': True,    # 保留对比度增强
            'enable_noise_reduction': True,         # 保留降噪
            'enable_perspective_correction': False, # 关闭透视校正，避免过度处理
            'enable_sharpening': True,              # 保留锐化，提高清晰度
            'enable_brightness_normalization': True, # 保留亮度标准化
            'target_height': 64,  # 目标高度
            'clahe_clip_limit': 1.5,  # 更温和的CLAHE强度，避免过度增强
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
        """自适应亮度和对比度标准化"""
        # 转换为灰度图进行分析
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 计算图像统计信息
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)
        median_brightness = np.median(gray)
        
        # 检测过度曝光（太亮）
        bright_pixel_ratio = np.sum(gray > 200) / gray.size
        very_bright_pixel_ratio = np.sum(gray > 240) / gray.size
        
        # 检测过度曝光的图像，减少或跳过处理
        if very_bright_pixel_ratio > 0.3 or mean_brightness > 200:
            self.logger.info(f"检测到过度曝光图像 (亮度均值: {mean_brightness:.1f}, 过亮像素比例: {very_bright_pixel_ratio:.3f})")
            # 对于过度曝光的图像，只进行轻微的对比度增强
            if std_contrast < 20:  # 只有对比度很低时才增强
                alpha = 1.1  # 轻微增强对比度
                beta = -10   # 轻微降低亮度
                enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                # 确保不超出范围
                enhanced = np.clip(enhanced, 0, 255)
                return enhanced
            else:
                return image  # 直接返回原图
        
        # 检测过暗图像
        dark_pixel_ratio = np.sum(gray < 50) / gray.size
        if mean_brightness < 80 and dark_pixel_ratio > 0.4:
            self.logger.info(f"检测到过暗图像 (亮度均值: {mean_brightness:.1f})")
            # 对于过暗图像，增加亮度
            target_brightness = 120
            brightness_factor = target_brightness - mean_brightness
            # 限制亮度调整幅度
            brightness_factor = np.clip(brightness_factor, 0, 60)
        else:
            # 正常图像的自适应调整
            if mean_brightness > 160:
                # 偏亮图像，目标亮度设低一些
                target_brightness = max(140, mean_brightness - 20)
            elif mean_brightness < 100:
                # 偏暗图像，目标亮度设高一些
                target_brightness = min(120, mean_brightness + 30)
            else:
                # 中等亮度图像
                target_brightness = 128
            
            brightness_factor = target_brightness - mean_brightness
            # 限制调整幅度，避免过度调整
            brightness_factor = np.clip(brightness_factor, -30, 40)
        
        # 自适应对比度调整
        if std_contrast < 20:
            # 低对比度，需要增强
            target_contrast = 40
        elif std_contrast > 80:
            # 高对比度，轻微增强或保持
            target_contrast = min(std_contrast * 1.1, 90)
        else:
            # 中等对比度
            target_contrast = 50
        
        # 计算对比度因子
        if std_contrast > 0:
            contrast_factor = target_contrast / std_contrast
            # 限制对比度调整幅度
            contrast_factor = np.clip(contrast_factor, 0.8, 2.0)
        else:
            contrast_factor = 1.0
        
        # 应用调整
        enhanced = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=brightness_factor)
        
        # 最终安全检查，防止过度曝光
        if len(enhanced.shape) == 3:
            check_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            check_gray = enhanced
            
        final_bright_ratio = np.sum(check_gray > 240) / check_gray.size
        if final_bright_ratio > 0.2:
            # 如果处理后仍有太多过亮像素，混合原图
            alpha = 0.7
            enhanced = cv2.addWeighted(image, 1-alpha, enhanced, alpha, 0)
            self.logger.info("检测到处理后过度曝光，与原图混合")
        
        return enhanced
    
    def _enhance_contrast(self, image: np.ndarray, level: str) -> np.ndarray:
        """自适应对比度增强"""
        # 分析图像亮度分布
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        mean_brightness = np.mean(gray)
        std_contrast = np.std(gray)
        very_bright_pixel_ratio = np.sum(gray > 240) / gray.size
        
        # 基础clip_limit值
        base_clip_limits = {
            'light': 1.5,
            'medium': 2.0,
            'strong': 2.5
        }
        base_clip_limit = base_clip_limits.get(level, 2.0)
        
        # 根据图像特征自适应调整clip_limit
        if very_bright_pixel_ratio > 0.2 or mean_brightness > 180:
            # 过亮图像：降低clip_limit，避免过度增强
            clip_limit = base_clip_limit * 0.5
            self.logger.info(f"过亮图像，降低CLAHE强度至 {clip_limit:.2f}")
        elif mean_brightness < 80:
            # 过暗图像：可以使用更高的clip_limit
            clip_limit = base_clip_limit * 1.3
            self.logger.info(f"过暗图像，提高CLAHE强度至 {clip_limit:.2f}")
        elif std_contrast < 20:
            # 低对比度图像：需要更强的增强
            clip_limit = base_clip_limit * 1.2
        else:
            # 正常图像
            clip_limit = base_clip_limit
        
        # 确保clip_limit在合理范围内
        clip_limit = np.clip(clip_limit, 0.5, 4.0)
        
        if len(image.shape) == 3:
            # 彩色图像，在LAB颜色空间中处理
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # 对于过亮图像，减小tile_grid_size以获得更局部的处理
            if very_bright_pixel_ratio > 0.2:
                tile_grid_size = (16, 16)  # 更小的tile获得更精细的控制
            else:
                tile_grid_size = (8, 8)
            
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_channel = lab[:,:,0]
            
            # 对于过亮图像，只在暗区域应用CLAHE
            if very_bright_pixel_ratio > 0.2:
                # 创建掩膜，只在非过亮区域应用增强
                mask = gray < 200
                enhanced_l = l_channel.copy()
                if np.any(mask):
                    enhanced_l[mask] = clahe.apply(l_channel)[mask]
                lab[:,:,0] = enhanced_l
            else:
                lab[:,:,0] = clahe.apply(l_channel)
            
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度图像
            if very_bright_pixel_ratio > 0.2:
                tile_grid_size = (16, 16)
            else:
                tile_grid_size = (8, 8)
                
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            if very_bright_pixel_ratio > 0.2:
                # 只在非过亮区域应用增强
                mask = gray < 200
                enhanced = image.copy()
                if np.any(mask):
                    enhanced[mask] = clahe.apply(image)[mask]
            else:
                enhanced = clahe.apply(image)
        
        # 后处理：检查是否产生过度增强
        if len(enhanced.shape) == 3:
            check_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            check_gray = enhanced
            
        final_bright_ratio = np.sum(check_gray > 245) / check_gray.size
        if final_bright_ratio > 0.15:
            # 如果增强后产生太多极亮像素，与原图混合
            alpha = 0.6
            enhanced = cv2.addWeighted(image, 1-alpha, enhanced, alpha, 0)
            self.logger.info("CLAHE处理后检测到过度增强，与原图混合")
        
        return enhanced
    
    def _sharpen_image(self, image: np.ndarray, level: str) -> np.ndarray:
        """图像锐化 - 针对车牌识别优化的温和锐化"""
        # 使用更温和的锐化核，避免过度锐化产生噪声
        kernels = {
            'light': np.array([[0, -0.3, 0], [-0.3, 2.2, -0.3], [0, -0.3, 0]]),
            'medium': np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]),
            'strong': np.array([[-0.5, -0.5, -0.5], [-0.5, 5, -0.5], [-0.5, -0.5, -0.5]])
        }
        
        kernel = kernels.get(level, kernels['medium'])
        enhanced = cv2.filter2D(image, -1, kernel)
        
        # 混合原图和锐化图，避免过度锐化
        alpha = 0.8  # 锐化强度
        enhanced = cv2.addWeighted(image, 1-alpha, enhanced, alpha, 0)
        
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

    def batch_enhance_directory(self, source_dir: str, target_dir: str, 
                              enhancement_level: str = 'medium',
                              supported_formats: List[str] = None) -> dict:
        """
        批量增强目录下的所有图像文件
        
        Args:
            source_dir: 源图像目录路径
            target_dir: 目标保存目录路径
            enhancement_level: 增强级别 ('light', 'medium', 'strong')
            supported_formats: 支持的图像格式列表，默认为常见图像格式
            
        Returns:
            处理结果统计字典
        """
        if supported_formats is None:
            supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
        
        # 确保目录存在
        source_path = Path(source_dir)
        target_path = Path(target_dir)
        
        if not source_path.exists():
            raise ValueError(f"源目录不存在: {source_dir}")
        
        # 创建目标目录
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有支持的图像文件
        image_files = []
        for fmt in supported_formats:
            image_files.extend(source_path.glob(f"*.{fmt}"))
            image_files.extend(source_path.glob(f"*.{fmt.upper()}"))
        
        if not image_files:
            self.logger.warning(f"在目录 {source_dir} 中未找到支持的图像文件")
            return {"success_count": 0, "error_count": 0, "total_count": 0}
        
        # 生成增强方法后缀
        enhancement_suffix = self._generate_enhancement_suffix()
        
        success_count = 0
        error_count = 0
        total_count = len(image_files)
        
        self.logger.info(f"开始批量处理 {total_count} 个图像文件")
        self.logger.info(f"增强配置后缀: {enhancement_suffix}")
        
        for img_file in image_files:
            try:
                # 读取图像
                image = cv2.imread(str(img_file))
                if image is None:
                    self.logger.error(f"无法读取图像文件: {img_file}")
                    error_count += 1
                    continue
                
                # 应用增强
                enhanced_image = self.enhance_plate_image(image, enhancement_level)
                
                # 生成新文件名
                original_name = img_file.stem  # 不包含扩展名的文件名
                extension = img_file.suffix     # 扩展名
                new_filename = f"{original_name}_{enhancement_suffix}{extension}"
                output_path = target_path / new_filename
                
                # 保存增强后的图像
                success = cv2.imwrite(str(output_path), enhanced_image)
                
                if success:
                    success_count += 1
                    self.logger.info(f"成功处理: {img_file.name} -> {new_filename}")
                else:
                    error_count += 1
                    self.logger.error(f"保存失败: {output_path}")
                    
            except Exception as e:
                error_count += 1
                self.logger.error(f"处理文件 {img_file} 时发生错误: {e}")
        
        # 记录处理结果
        result = {
            "success_count": success_count,
            "error_count": error_count,
            "total_count": total_count
        }
        
        self.logger.info(f"批量处理完成: 成功 {success_count}/{total_count} 个文件")
        
        return result
    
    def _generate_enhancement_suffix(self) -> str:
        """
        根据当前配置生成增强方法后缀
        
        Returns:
            增强方法的后缀字符串
        """
        enabled_methods = []
        
        if self.config.get('enable_noise_reduction', False):
            enabled_methods.append('noise_reduction')
        if self.config.get('enable_contrast_enhancement', False):
            enabled_methods.append('contrast')
        if self.config.get('enable_sharpening', False):
            enabled_methods.append('sharpening')
        if self.config.get('enable_adaptive_thresholding', False):
            enabled_methods.append('adaptive_thresh')
        if self.config.get('enable_perspective_correction', False):
            enabled_methods.append('perspective')
        if self.config.get('enable_brightness_normalization', False):
            enabled_methods.append('brightness_norm')
        
        if not enabled_methods:
            return 'no_enhancement'
        
        # 如果启用的方法太多，使用简化后缀
        if len(enabled_methods) > 3:
            return 'multi_enhancement'
        
        return '_'.join(enabled_methods)
    
    def batch_enhance_with_multiple_levels(self, source_dir: str, target_dir: str,
                                         enhancement_levels: List[str] = None,
                                         supported_formats: List[str] = None) -> dict:
        """
        使用多个增强级别批量处理图像
        
        Args:
            source_dir: 源图像目录路径
            target_dir: 目标保存目录路径
            enhancement_levels: 增强级别列表，默认为 ['light', 'medium', 'strong']
            supported_formats: 支持的图像格式列表
            
        Returns:
            处理结果统计字典
        """
        if enhancement_levels is None:
            enhancement_levels = ['light', 'medium', 'strong']
        
        total_results = {
            "success_count": 0,
            "error_count": 0,
            "total_count": 0,
            "level_results": {}
        }
        
        for level in enhancement_levels:
            self.logger.info(f"开始处理增强级别: {level}")
            
            # 为每个级别创建子目录
            level_target_dir = Path(target_dir) / level
            
            try:
                result = self.batch_enhance_directory(
                    source_dir, 
                    str(level_target_dir), 
                    level, 
                    supported_formats
                )
                
                total_results["level_results"][level] = result
                total_results["success_count"] += result["success_count"]
                total_results["error_count"] += result["error_count"]
                
                # total_count只计算一次
                if total_results["total_count"] == 0:
                    total_results["total_count"] = result["total_count"]
                    
            except Exception as e:
                self.logger.error(f"处理增强级别 {level} 时发生错误: {e}")
                total_results["level_results"][level] = {
                    "success_count": 0,
                    "error_count": 0,
                    "total_count": 0,
                    "error": str(e)
                }
        
        return total_results 