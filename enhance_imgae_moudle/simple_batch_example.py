#!/usr/bin/env python3
"""
简单的批量数据增强使用示例
"""

import logging
from plate_enhancer import PlateEnhancer

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def example_basic_batch_enhance():
    """基础批量增强示例"""
    print("=== 基础批量增强示例 ===")
    
    # 创建增强器（使用默认配置）
    enhancer = PlateEnhancer()
    
    # 批量处理目录
    source_dir = "input_images"  # 替换为你的源图像目录
    target_dir = "enhanced_images"  # 替换为你的目标目录
    
    try:
        results = enhancer.batch_enhance_directory(
            source_dir=source_dir,
            target_dir=target_dir,
            enhancement_level='medium'
        )
        
        print(f"处理结果: 成功 {results['success_count']}/{results['total_count']} 个文件")
        
    except Exception as e:
        print(f"处理失败: {e}")

def example_custom_config_batch():
    """自定义配置批量增强示例"""
    print("\n=== 自定义配置批量增强示例 ===")
    
    # 自定义配置：只启用噪声减少和对比度增强
    custom_config = {
        'enable_noise_reduction': True,
        'enable_contrast_enhancement': True,
        'enable_sharpening': False,
        'enable_adaptive_thresholding': False,
        'enable_perspective_correction': False,
        'enable_brightness_normalization': True,
        'target_height': 64,
    }
    
    # 创建增强器
    enhancer = PlateEnhancer(custom_config)
    
    source_dir = "input_images"
    target_dir = "custom_enhanced_images"
    
    try:
        results = enhancer.batch_enhance_directory(
            source_dir=source_dir,
            target_dir=target_dir,
            enhancement_level='strong'
        )
        
        print(f"处理结果: 成功 {results['success_count']}/{results['total_count']} 个文件")
        print("增强后的文件名会包含后缀: noise_reduction_contrast_brightness_norm")
        
    except Exception as e:
        print(f"处理失败: {e}")

def example_multiple_levels():
    """多级别增强示例"""
    print("\n=== 多级别增强示例 ===")
    
    # 创建增强器
    enhancer = PlateEnhancer()
    
    source_dir = "input_images"
    target_dir = "multi_level_enhanced"
    
    try:
        results = enhancer.batch_enhance_with_multiple_levels(
            source_dir=source_dir,
            target_dir=target_dir,
            enhancement_levels=['light', 'medium', 'strong']
        )
        
        print(f"总处理结果: 成功 {results['success_count']}/{results['total_count']} 个文件")
        
        # 显示各级别的详细结果
        for level, level_result in results['level_results'].items():
            print(f"  级别 {level}: 成功 {level_result['success_count']}/{level_result['total_count']}")
        
        print("结果会保存在以下子目录中:")
        print("  - multi_level_enhanced/light/")
        print("  - multi_level_enhanced/medium/")  
        print("  - multi_level_enhanced/strong/")
        
    except Exception as e:
        print(f"处理失败: {e}")

def example_specific_formats():
    """指定图像格式示例"""
    print("\n=== 指定图像格式示例 ===")
    
    enhancer = PlateEnhancer()
    
    source_dir = "input_images"
    target_dir = "jpg_enhanced"
    
    try:
        # 只处理 JPG 格式的图像
        results = enhancer.batch_enhance_directory(
            source_dir=source_dir,
            target_dir=target_dir,
            enhancement_level='medium',
            supported_formats=['jpg', 'jpeg']  # 只处理JPG格式
        )
        
        print(f"JPG图像处理结果: 成功 {results['success_count']}/{results['total_count']} 个文件")
        
    except Exception as e:
        print(f"处理失败: {e}")

if __name__ == '__main__':
    print("车牌图像批量增强示例")
    print("注意：请确保源目录存在并包含图像文件")
    
    # 运行示例（取消注释需要运行的示例）
    example_basic_batch_enhance()
    # example_custom_config_batch()
    # example_multiple_levels()
    # example_specific_formats()
    
    print("\n完成！") 