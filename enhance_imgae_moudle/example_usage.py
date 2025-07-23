"""
车牌图像增强模块使用示例
展示如何使用不同的增强功能来提高车牌识别准确性
"""

import cv2
import numpy as np
import os
import logging
from typing import List

# 导入增强模块
from plate_enhancer import PlateEnhancer
from image_processor import ImageProcessor
from noise_reducer import NoiseReducer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_images(image_dir: str) -> List[np.ndarray]:
    """加载测试图像"""
    images = []
    if not os.path.exists(image_dir):
        logger.warning(f"目录不存在: {image_dir}")
        return images
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(image_dir, filename)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
                logger.info(f"加载图像: {filename}")
    
    return images

def demonstrate_basic_enhancement():
    """演示基本增强功能"""
    print("\n=== 基本车牌图像增强演示 ===")
    
    # 创建车牌增强器
    enhancer = PlateEnhancer()
    
    # 创建一个示例图像（如果没有真实图像）
    # 这里创建一个模拟的车牌图像
    demo_image = create_demo_plate_image()
    
    # 不同级别的增强
    enhanced_light = enhancer.enhance_plate_image(demo_image, 'light')
    enhanced_medium = enhancer.enhance_plate_image(demo_image, 'medium')
    enhanced_strong = enhancer.enhance_plate_image(demo_image, 'strong')
    
    # 显示结果（如果有显示环境）
    try:
        cv2.imshow('原始图像', demo_image)
        cv2.imshow('轻度增强', enhanced_light)
        cv2.imshow('中度增强', enhanced_medium)
        cv2.imshow('强度增强', enhanced_strong)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        logger.info(f"无法显示图像: {e}")
        # 保存到文件
        cv2.imwrite('demo_original.jpg', demo_image)
        cv2.imwrite('demo_enhanced_light.jpg', enhanced_light)
        cv2.imwrite('demo_enhanced_medium.jpg', enhanced_medium)
        cv2.imwrite('demo_enhanced_strong.jpg', enhanced_strong)
        print("增强结果已保存到文件")

def demonstrate_noise_reduction():
    """演示噪声减少功能"""
    print("\n=== 噪声减少演示 ===")
    
    noise_reducer = NoiseReducer()
    
    # 创建带噪声的示例图像
    demo_image = create_demo_plate_image()
    noisy_image = add_noise_to_image(demo_image)
    
    # 不同的去噪方法
    methods = ['gaussian', 'bilateral', 'median', 'non_local_means']
    
    for method in methods:
        try:
            denoised = noise_reducer.reduce_noise(noisy_image, method, 'medium')
            filename = f'denoised_{method}.jpg'
            cv2.imwrite(filename, denoised)
            print(f"{method} 去噪结果保存为: {filename}")
        except Exception as e:
            logger.error(f"{method} 去噪失败: {e}")

def demonstrate_image_processing():
    """演示图像处理功能"""
    print("\n=== 图像处理演示 ===")
    
    processor = ImageProcessor()
    demo_image = create_demo_plate_image()
    
    # 倾斜校正
    skewed_image = rotate_image(demo_image, 15)  # 模拟倾斜
    corrected_image, angle = processor.correct_skew(skewed_image)
    print(f"检测到倾斜角度: {angle:.2f}度")
    
    # 对比度增强
    enhanced_contrast = processor.enhance_local_contrast(demo_image)
    
    # 边缘增强
    enhanced_edges = processor.enhance_edges(demo_image, 'sobel')
    
    # 保存结果
    cv2.imwrite('skewed_image.jpg', skewed_image)
    cv2.imwrite('corrected_image.jpg', corrected_image)
    cv2.imwrite('enhanced_contrast.jpg', enhanced_contrast)
    cv2.imwrite('enhanced_edges.jpg', enhanced_edges)
    print("图像处理结果已保存")

def demonstrate_batch_processing():
    """演示批量处理"""
    print("\n=== 批量处理演示 ===")
    
    enhancer = PlateEnhancer()
    
    # 创建多个示例图像
    images = [create_demo_plate_image() for _ in range(3)]
    
    # 批量增强
    enhanced_images = enhancer.batch_enhance(images, 'medium')
    
    # 保存结果
    for i, enhanced in enumerate(enhanced_images):
        filename = f'batch_enhanced_{i+1}.jpg'
        cv2.imwrite(filename, enhanced)
        print(f"批量处理结果 {i+1} 保存为: {filename}")

def demonstrate_custom_config():
    """演示自定义配置"""
    print("\n=== 自定义配置演示 ===")
    
    # 自定义配置
    custom_config = {
        'enable_adaptive_thresholding': True,
        'enable_contrast_enhancement': True,
        'enable_noise_reduction': True,
        'enable_perspective_correction': False,  # 禁用透视校正
        'enable_sharpening': True,
        'enable_brightness_normalization': True,
        'target_height': 80,  # 更大的目标高度
        'clahe_clip_limit': 4.0,  # 更强的对比度增强
    }
    
    enhancer = PlateEnhancer(custom_config)
    demo_image = create_demo_plate_image()
    
    enhanced = enhancer.enhance_plate_image(demo_image, 'medium')
    cv2.imwrite('custom_enhanced.jpg', enhanced)
    print("自定义配置增强结果保存为: custom_enhanced.jpg")

def create_demo_plate_image() -> np.ndarray:
    """创建演示用的车牌图像"""
    # 创建一个白色背景的图像
    img = np.ones((60, 200, 3), dtype=np.uint8) * 255
    
    # 添加蓝色边框（模拟车牌）
    cv2.rectangle(img, (5, 5), (195, 55), (255, 0, 0), 2)
    
    # 添加黑色文字（模拟车牌号码）
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'ABC 123', (20, 35), font, 1, (0, 0, 0), 2)
    
    return img

def add_noise_to_image(image: np.ndarray) -> np.ndarray:
    """向图像添加噪声"""
    # 添加高斯噪声
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    
    # 添加椒盐噪声
    salt_pepper = np.random.random(image.shape[:2])
    noisy[salt_pepper < 0.05] = 0
    noisy[salt_pepper > 0.95] = 255
    
    return noisy

def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """旋转图像"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated

def benchmark_enhancement_methods():
    """性能基准测试"""
    print("\n=== 性能基准测试 ===")
    
    import time
    
    enhancer = PlateEnhancer()
    demo_image = create_demo_plate_image()
    
    # 测试不同增强级别的性能
    levels = ['light', 'medium', 'strong']
    
    for level in levels:
        start_time = time.time()
        for _ in range(10):  # 重复10次
            enhanced = enhancer.enhance_plate_image(demo_image, level)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000  # 转换为毫秒
        print(f"{level} 级别增强平均耗时: {avg_time:.2f}ms")

def main():
    """主函数"""
    print("车牌图像增强模块演示程序")
    print("=============================")
    
    try:
        # 基本功能演示
        demonstrate_basic_enhancement()
        
        # 噪声减少演示
        demonstrate_noise_reduction()
        
        # 图像处理演示
        demonstrate_image_processing()
        
        # 批量处理演示
        demonstrate_batch_processing()
        
        # 自定义配置演示
        demonstrate_custom_config()
        
        # 性能测试
        benchmark_enhancement_methods()
        
        print("\n演示完成！请查看生成的图像文件。")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")

if __name__ == "__main__":
    main() 