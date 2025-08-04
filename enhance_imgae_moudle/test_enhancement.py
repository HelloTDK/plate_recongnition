#!/usr/bin/env python3
"""
测试车牌图像增强效果的脚本
"""

import cv2
import numpy as np
import logging
from plate_enhancer import PlateEnhancer
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_image(image_path: str):
    """测试单张图像的增强效果"""
    print(f"\n=== 测试图像: {image_path} ===")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return
    
    print(f"原始图像尺寸: {image.shape}")
    
    # 创建增强器（车牌识别优化模式）
    enhancer = PlateEnhancer()
    
    # 应用增强
    enhanced = enhancer.enhance_plate_image(image, 'medium')
    
    print(f"增强后图像尺寸: {enhanced.shape}")
    print(f"增强后图像类型: {enhanced.dtype}")
    print(f"增强后图像值范围: {enhanced.min()} - {enhanced.max()}")
    
    # 检查是否为二值图
    unique_values = len(np.unique(enhanced))
    print(f"图像唯一值数量: {unique_values}")
    
    if unique_values <= 2:
        print("⚠️  警告：图像似乎被转换为二值图")
    else:
        print("✅ 良好：图像保持了颜色信息")
    
    # 保存结果用于检查
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    original_name = Path(image_path).stem
    cv2.imwrite(f"test_output/{original_name}_original.jpg", image)
    cv2.imwrite(f"test_output/{original_name}_enhanced.jpg", enhanced)
    
    print(f"结果已保存到 test_output/ 目录")

def test_different_levels():
    """测试不同增强级别"""
    print("\n=== 测试不同增强级别 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (64, 200, 3), dtype=np.uint8)
    
    enhancer = PlateEnhancer()
    levels = ['light', 'medium', 'strong']
    
    for level in levels:
        enhanced = enhancer.enhance_plate_image(test_image, level)
        unique_values = len(np.unique(enhanced))
        print(f"级别 {level}: 唯一值数量 = {unique_values}")
        
        # 保存测试结果
        cv2.imwrite(f"test_output/test_{level}.jpg", enhanced)

def test_brightness_adaptation():
    """测试亮度自适应功能"""
    print("\n=== 测试亮度自适应功能 ===")
    
    enhancer = PlateEnhancer()
    
    # 创建不同亮度的测试图像
    test_cases = {
        '过暗图像': np.random.randint(0, 80, (64, 200, 3), dtype=np.uint8),
        '正常图像': np.random.randint(50, 200, (64, 200, 3), dtype=np.uint8), 
        '偏亮图像': np.random.randint(150, 255, (64, 200, 3), dtype=np.uint8),
        '过亮图像': np.full((64, 200, 3), 240, dtype=np.uint8),  # 极亮图像
        '高对比度亮图': None,  # 后面创建
        '全白图像': np.full((64, 200, 3), 255, dtype=np.uint8),
    }
    
    # 创建高对比度亮图
    bright_contrast = np.full((64, 200, 3), 220, dtype=np.uint8)
    # 添加一些暗区域形成对比
    bright_contrast[20:40, 50:150] = 100
    test_cases['高对比度亮图'] = bright_contrast
    
    for test_name, test_image in test_cases.items():
        print(f"\n--- {test_name} ---")
        
        # 分析原图特征
        if len(test_image.shape) == 3:
            gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = test_image
            
        mean_bright = np.mean(gray)
        std_contrast = np.std(gray)
        very_bright_ratio = np.sum(gray > 240) / gray.size
        
        print(f"原图亮度均值: {mean_bright:.1f}")
        print(f"原图对比度(std): {std_contrast:.1f}")
        print(f"过亮像素比例: {very_bright_ratio:.3f}")
        
        # 应用增强
        enhanced = enhancer.enhance_plate_image(test_image, 'medium')
        
        # 分析增强后特征
        if len(enhanced.shape) == 3:
            enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_gray = enhanced
            
        enhanced_mean = np.mean(enhanced_gray)
        enhanced_std = np.std(enhanced_gray)
        enhanced_bright_ratio = np.sum(enhanced_gray > 240) / enhanced_gray.size
        white_ratio = np.sum(enhanced_gray > 250) / enhanced_gray.size
        
        print(f"增强后亮度均值: {enhanced_mean:.1f}")
        print(f"增强后对比度(std): {enhanced_std:.1f}")
        print(f"增强后过亮像素比例: {enhanced_bright_ratio:.3f}")
        print(f"增强后接近全白像素比例: {white_ratio:.3f}")
        
        # 判断处理效果
        if white_ratio > 0.5:
            print("❌ 警告：图像接近全白！")
        elif enhanced_bright_ratio > 0.8:
            print("⚠️  注意：图像过度曝光")
        else:
            print("✅ 正常：图像亮度适中")
        
        # 保存结果
        safe_name = test_name.replace(' ', '_')
        cv2.imwrite(f"test_output/{safe_name}_original.jpg", test_image)
        cv2.imwrite(f"test_output/{safe_name}_enhanced.jpg", enhanced)

def test_configuration():
    """测试不同配置"""
    print("\n=== 测试不同配置 ===")
    
    # 创建测试图像
    test_image = np.random.randint(0, 256, (64, 200, 3), dtype=np.uint8)
    
    configs = {
        '默认配置': {},
        '只降噪': {
            'enable_noise_reduction': True,
            'enable_contrast_enhancement': False,
            'enable_sharpening': False,
            'enable_adaptive_thresholding': False,
            'enable_perspective_correction': False,
            'enable_brightness_normalization': False,
        },
        '只清晰化': {
            'enable_noise_reduction': False,
            'enable_contrast_enhancement': True,
            'enable_sharpening': True,
            'enable_adaptive_thresholding': False,
            'enable_perspective_correction': False,
            'enable_brightness_normalization': False,
        }
    }
    
    for config_name, config in configs.items():
        enhancer = PlateEnhancer(config)
        enhanced = enhancer.enhance_plate_image(test_image, 'medium')
        unique_values = len(np.unique(enhanced))
        print(f"{config_name}: 唯一值数量 = {unique_values}")

if __name__ == '__main__':
    print("车牌图像增强测试")
    
    # 创建输出目录
    Path("test_output").mkdir(exist_ok=True)
    
    # 测试不同级别
    test_different_levels()
    
    # 重点测试亮度自适应功能
    test_brightness_adaptation()
    
    # 测试不同配置
    test_configuration()
    
    # 如果有真实图像，可以测试
    test_images = [
        "sample_plate.jpg",
        "test_image.png",
        # 添加你想测试的图像路径
    ]
    
    for img_path in test_images:
        if Path(img_path).exists():
            test_single_image(img_path)
        else:
            print(f"跳过不存在的测试图像: {img_path}")
    
    print("\n测试完成！检查 test_output/ 目录中的结果")
    print("特别关注亮度自适应测试结果，确认过亮图像没有变成全白") 