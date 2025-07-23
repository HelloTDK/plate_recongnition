"""
噪声减少模块
提供多种去噪算法，专门针对车牌图像优化
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

class NoiseReducer:
    """
    噪声减少器
    提供多种去噪算法来清理车牌图像
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def reduce_noise(self, image: np.ndarray, method: str = 'bilateral', 
                    strength: str = 'medium') -> np.ndarray:
        """
        减少图像噪声
        
        Args:
            image: 输入图像
            method: 去噪方法 ('gaussian', 'bilateral', 'median', 'non_local_means', 'morphological')
            strength: 去噪强度 ('light', 'medium', 'strong')
            
        Returns:
            去噪后的图像
        """
        if method == 'gaussian':
            return self._gaussian_denoising(image, strength)
        elif method == 'bilateral':
            return self._bilateral_denoising(image, strength)
        elif method == 'median':
            return self._median_denoising(image, strength)
        elif method == 'non_local_means':
            return self._non_local_means_denoising(image, strength)
        elif method == 'morphological':
            return self._morphological_denoising(image, strength)
        else:
            raise ValueError(f"未支持的去噪方法: {method}")
    
    def _gaussian_denoising(self, image: np.ndarray, strength: str) -> np.ndarray:
        """高斯去噪"""
        kernel_sizes = {
            'light': (3, 3),
            'medium': (5, 5),
            'strong': (7, 7)
        }
        
        sigma_values = {
            'light': 0.5,
            'medium': 1.0,
            'strong': 1.5
        }
        
        kernel_size = kernel_sizes.get(strength, (5, 5))
        sigma = sigma_values.get(strength, 1.0)
        
        denoised = cv2.GaussianBlur(image, kernel_size, sigma)
        return denoised
    
    def _bilateral_denoising(self, image: np.ndarray, strength: str) -> np.ndarray:
        """双边滤波去噪"""
        params = {
            'light': {'d': 5, 'sigma_color': 50, 'sigma_space': 50},
            'medium': {'d': 9, 'sigma_color': 75, 'sigma_space': 75},
            'strong': {'d': 13, 'sigma_color': 100, 'sigma_space': 100}
        }
        
        param = params.get(strength, params['medium'])
        
        denoised = cv2.bilateralFilter(
            image, 
            param['d'], 
            param['sigma_color'], 
            param['sigma_space']
        )
        
        return denoised
    
    def _median_denoising(self, image: np.ndarray, strength: str) -> np.ndarray:
        """中值滤波去噪"""
        kernel_sizes = {
            'light': 3,
            'medium': 5,
            'strong': 7
        }
        
        kernel_size = kernel_sizes.get(strength, 5)
        
        denoised = cv2.medianBlur(image, kernel_size)
        return denoised
    
    def _non_local_means_denoising(self, image: np.ndarray, strength: str) -> np.ndarray:
        """非局部均值去噪"""
        params = {
            'light': {'h': 3, 'template_window_size': 7, 'search_window_size': 21},
            'medium': {'h': 5, 'template_window_size': 7, 'search_window_size': 21},
            'strong': {'h': 10, 'template_window_size': 9, 'search_window_size': 25}
        }
        
        param = params.get(strength, params['medium'])
        
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                param['h'],
                param['h'],
                param['template_window_size'],
                param['search_window_size']
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                image,
                None,
                param['h'],
                param['template_window_size'],
                param['search_window_size']
            )
        
        return denoised
    
    def _morphological_denoising(self, image: np.ndarray, strength: str) -> np.ndarray:
        """形态学去噪"""
        kernel_sizes = {
            'light': (2, 2),
            'medium': (3, 3),
            'strong': (4, 4)
        }
        
        kernel_size = kernel_sizes.get(strength, (3, 3))
        kernel = np.ones(kernel_size, np.uint8)
        
        # 开运算去除小噪点
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # 闭运算填补小孔
        denoised = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return denoised
    
    def remove_salt_pepper_noise(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        去除椒盐噪声
        
        Args:
            image: 输入图像
            kernel_size: 中值滤波核大小
            
        Returns:
            去除椒盐噪声后的图像
        """
        return cv2.medianBlur(image, kernel_size)
    
    def remove_gaussian_noise(self, image: np.ndarray, kernel_size: Tuple[int, int] = (5, 5), 
                             sigma: float = 1.0) -> np.ndarray:
        """
        去除高斯噪声
        
        Args:
            image: 输入图像
            kernel_size: 高斯核大小
            sigma: 标准差
            
        Returns:
            去除高斯噪声后的图像
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def adaptive_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        自适应去噪
        根据图像特性自动选择最佳去噪参数
        
        Args:
            image: 输入图像
            
        Returns:
            去噪后的图像
        """
        # 计算图像噪声水平
        noise_level = self._estimate_noise_level(image)
        
        # 根据噪声水平选择去噪策略
        if noise_level < 10:
            # 低噪声，使用轻度去噪
            return self._bilateral_denoising(image, 'light')
        elif noise_level < 20:
            # 中等噪声，使用中度去噪
            return self._bilateral_denoising(image, 'medium')
        else:
            # 高噪声，使用强度去噪
            return self._non_local_means_denoising(image, 'medium')
    
    def _estimate_noise_level(self, image: np.ndarray) -> float:
        """
        估计图像噪声水平
        
        Args:
            image: 输入图像
            
        Returns:
            噪声水平估计值
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 使用拉普拉斯算子估计噪声
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = laplacian.var()
        
        return noise_level
    
    def wiener_filter(self, image: np.ndarray, noise_variance: Optional[float] = None) -> np.ndarray:
        """
        维纳滤波去噪
        
        Args:
            image: 输入图像
            noise_variance: 噪声方差，如果为None则自动估计
            
        Returns:
            滤波后的图像
        """
        if len(image.shape) == 3:
            # 对彩色图像的每个通道分别处理
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = self._apply_wiener_filter(image[:, :, i], noise_variance)
            return result
        else:
            return self._apply_wiener_filter(image, noise_variance)
    
    def _apply_wiener_filter(self, channel: np.ndarray, noise_variance: Optional[float]) -> np.ndarray:
        """对单通道应用维纳滤波"""
        # 转换为浮点数
        f_image = channel.astype(np.float32)
        
        # FFT变换
        f_transform = np.fft.fft2(f_image)
        
        if noise_variance is None:
            # 估计噪声方差
            noise_variance = np.var(f_image) * 0.1
        
        # 计算功率谱
        power_spectrum = np.abs(f_transform) ** 2
        
        # 维纳滤波
        wiener_filter = power_spectrum / (power_spectrum + noise_variance)
        filtered_transform = f_transform * wiener_filter
        
        # 逆FFT变换
        filtered_image = np.fft.ifft2(filtered_transform)
        filtered_image = np.real(filtered_image)
        
        # 确保值在有效范围内
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
        
        return filtered_image
    
    def anisotropic_diffusion(self, image: np.ndarray, iterations: int = 10, 
                             kappa: float = 20, gamma: float = 0.1) -> np.ndarray:
        """
        各向异性扩散去噪
        
        Args:
            image: 输入图像
            iterations: 迭代次数
            kappa: 传导系数
            gamma: 积分常数
            
        Returns:
            去噪后的图像
        """
        if len(image.shape) == 3:
            # 对彩色图像的每个通道分别处理
            result = np.zeros_like(image, dtype=np.float32)
            for i in range(3):
                result[:, :, i] = self._anisotropic_diffusion_channel(
                    image[:, :, i].astype(np.float32), iterations, kappa, gamma
                )
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            result = self._anisotropic_diffusion_channel(
                image.astype(np.float32), iterations, kappa, gamma
            )
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def _anisotropic_diffusion_channel(self, channel: np.ndarray, iterations: int, 
                                     kappa: float, gamma: float) -> np.ndarray:
        """对单通道应用各向异性扩散"""
        img = channel.copy()
        
        for _ in range(iterations):
            # 计算梯度
            grad_n = np.roll(img, -1, axis=0) - img  # 北方向
            grad_s = np.roll(img, 1, axis=0) - img   # 南方向
            grad_e = np.roll(img, -1, axis=1) - img  # 东方向
            grad_w = np.roll(img, 1, axis=1) - img   # 西方向
            
            # 计算传导系数
            c_n = np.exp(-((grad_n / kappa) ** 2))
            c_s = np.exp(-((grad_s / kappa) ** 2))
            c_e = np.exp(-((grad_e / kappa) ** 2))
            c_w = np.exp(-((grad_w / kappa) ** 2))
            
            # 更新图像
            img += gamma * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w)
        
        return img
    
    def combine_denoising_methods(self, image: np.ndarray, methods: list = None) -> np.ndarray:
        """
        组合多种去噪方法
        
        Args:
            image: 输入图像
            methods: 去噪方法列表，如果为None则使用默认组合
            
        Returns:
            组合去噪后的图像
        """
        if methods is None:
            methods = ['median', 'bilateral', 'gaussian']
        
        result = image.copy()
        
        for method in methods:
            try:
                result = self.reduce_noise(result, method, 'light')
                self.logger.info(f"应用 {method} 去噪方法")
            except Exception as e:
                self.logger.warning(f"应用 {method} 去噪方法时出错: {e}")
                continue
        
        return result 