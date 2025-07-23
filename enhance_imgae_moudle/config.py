"""
配置管理模块
提供不同场景下的预设配置和自定义配置管理
"""

from typing import Dict, Any
import json
import os

class EnhancementConfig:
    """
    增强配置管理类
    提供不同场景的预设配置
    """
    
    # 预设配置
    PRESETS = {
        'low_light': {
            'description': '低光照环境车牌增强',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 64,
            'clahe_clip_limit': 4.0,
            'gaussian_kernel_size': 3,
            'bilateral_d': 9,
            'bilateral_sigma_color': 100,
            'bilateral_sigma_space': 100,
            'brightness_adjustment': 30,
            'contrast_adjustment': 1.2
        },
        
        'high_noise': {
            'description': '高噪声环境车牌增强',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': False,  # 高噪声时避免锐化
            'enable_brightness_normalization': True,
            'target_height': 64,
            'clahe_clip_limit': 2.0,
            'gaussian_kernel_size': 5,
            'bilateral_d': 13,
            'bilateral_sigma_color': 150,
            'bilateral_sigma_space': 150,
            'noise_reduction_strength': 'strong'
        },
        
        'motion_blur': {
            'description': '运动模糊车牌增强',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': False,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 64,
            'clahe_clip_limit': 3.0,
            'sharpening_strength': 'strong',
            'deblur_kernel_size': 15,
            'deblur_angle': 0
        },
        
        'perspective_distortion': {
            'description': '透视畸变车牌增强',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 64,
            'perspective_correction_aggressive': True,
            'edge_threshold_low': 30,
            'edge_threshold_high': 100
        },
        
        'weather_condition': {
            'description': '恶劣天气条件车牌增强',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'enable_shadow_removal': True,
            'target_height': 64,
            'clahe_clip_limit': 3.5,
            'weather_enhancement': True,
            'haze_removal': True
        },
        
        'high_resolution': {
            'description': '高分辨率车牌增强',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 128,  # 更高的目标分辨率
            'clahe_clip_limit': 2.5,
            'preserve_detail': True,
            'super_resolution': True
        },
        
        'real_time': {
            'description': '实时处理优化配置',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': False,  # 关闭耗时的去噪
            'enable_perspective_correction': False,  # 关闭透视校正
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 48,  # 较小的尺寸以提高速度
            'clahe_clip_limit': 2.0,
            'fast_mode': True
        },
        
        'default': {
            'description': '默认通用配置',
            'enable_adaptive_thresholding': True,
            'enable_contrast_enhancement': True,
            'enable_noise_reduction': True,
            'enable_perspective_correction': True,
            'enable_sharpening': True,
            'enable_brightness_normalization': True,
            'target_height': 64,
            'clahe_clip_limit': 3.0,
            'gaussian_kernel_size': 3,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75
        }
    }
    
    @classmethod
    def get_preset(cls, preset_name: str) -> Dict[str, Any]:
        """
        获取预设配置
        
        Args:
            preset_name: 预设名称
            
        Returns:
            配置字典
        """
        if preset_name not in cls.PRESETS:
            raise ValueError(f"未知的预设配置: {preset_name}")
        
        return cls.PRESETS[preset_name].copy()
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """
        列出所有可用的预设配置
        
        Returns:
            预设名称和描述的字典
        """
        return {name: config['description'] for name, config in cls.PRESETS.items()}
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], file_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            file_path: 文件路径
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_config(cls, file_path: str) -> Dict[str, Any]:
        """
        从文件加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            配置字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def merge_configs(cls, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置，override_config 中的值会覆盖 base_config
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """
        验证配置的有效性
        
        Args:
            config: 配置字典
            
        Returns:
            是否有效
        """
        required_keys = [
            'enable_adaptive_thresholding',
            'enable_contrast_enhancement',
            'enable_noise_reduction',
            'enable_perspective_correction',
            'enable_sharpening',
            'enable_brightness_normalization',
            'target_height'
        ]
        
        # 检查必需的键
        for key in required_keys:
            if key not in config:
                return False
        
        # 检查数值范围
        if config.get('target_height', 0) <= 0:
            return False
        
        if config.get('clahe_clip_limit', 0) <= 0:
            return False
        
        return True
    
    @classmethod
    def create_adaptive_config(cls, image_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据图像特征创建自适应配置
        
        Args:
            image_characteristics: 图像特征字典
                - brightness: 亮度级别 (0-255)
                - noise_level: 噪声级别 (0-100)
                - blur_level: 模糊级别 (0-100)
                - contrast: 对比度 (0-100)
                
        Returns:
            自适应配置
        """
        base_config = cls.get_preset('default')
        
        brightness = image_characteristics.get('brightness', 128)
        noise_level = image_characteristics.get('noise_level', 10)
        blur_level = image_characteristics.get('blur_level', 0)
        contrast = image_characteristics.get('contrast', 50)
        
        # 根据亮度调整
        if brightness < 80:
            # 低亮度
            base_config.update({
                'clahe_clip_limit': 4.0,
                'enable_brightness_normalization': True,
                'brightness_adjustment': 30
            })
        elif brightness > 180:
            # 高亮度
            base_config.update({
                'clahe_clip_limit': 2.0,
                'brightness_adjustment': -20
            })
        
        # 根据噪声级别调整
        if noise_level > 30:
            base_config.update({
                'enable_noise_reduction': True,
                'bilateral_d': 13,
                'bilateral_sigma_color': 150,
                'bilateral_sigma_space': 150,
                'enable_sharpening': False  # 高噪声时避免锐化
            })
        
        # 根据模糊级别调整
        if blur_level > 20:
            base_config.update({
                'enable_sharpening': True,
                'sharpening_strength': 'strong'
            })
        
        # 根据对比度调整
        if contrast < 30:
            base_config.update({
                'clahe_clip_limit': 4.0,
                'enable_contrast_enhancement': True
            })
        
        return base_config

# 使用示例配置
def create_example_configs():
    """创建示例配置文件"""
    config_manager = EnhancementConfig()
    
    # 保存所有预设配置到文件
    for preset_name in config_manager.PRESETS.keys():
        config = config_manager.get_preset(preset_name)
        filename = f"config_{preset_name}.json"
        config_manager.save_config(config, filename)
        print(f"已保存配置: {filename}")

if __name__ == "__main__":
    # 演示配置管理功能
    print("车牌增强配置管理演示")
    print("====================")
    
    config_manager = EnhancementConfig()
    
    # 列出所有预设
    print("\n可用的预设配置:")
    for name, description in config_manager.list_presets().items():
        print(f"  {name}: {description}")
    
    # 获取低光照配置
    low_light_config = config_manager.get_preset('low_light')
    print(f"\n低光照配置: {low_light_config}")
    
    # 创建自适应配置
    image_chars = {
        'brightness': 60,
        'noise_level': 25,
        'blur_level': 15,
        'contrast': 40
    }
    adaptive_config = config_manager.create_adaptive_config(image_chars)
    print(f"\n自适应配置: {adaptive_config}")
    
    # 创建示例配置文件
    create_example_configs() 