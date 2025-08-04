#!/usr/bin/env python3
"""
批量数据增强示例脚本
使用PlateEnhancer进行批量图像增强处理
"""

import logging
import argparse
import sys
from pathlib import Path
from plate_enhancer import PlateEnhancer

def setup_logging(log_level: str = 'INFO'):
    """设置日志配置"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('batch_enhance.log', encoding='utf-8')
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量车牌图像增强工具')
    parser.add_argument('--source_dir', default='/expdata/givap/data/plate_recong/mix/mix_b1',help='源图像目录路径')
    parser.add_argument('--target_dir', default='/expdata/givap/data/plate_recong/mix/mix_b1_enhance', help='目标保存目录路径')
    parser.add_argument('--level', choices=['light', 'medium', 'strong'], 
                       default='medium', help='增强级别 (默认: medium)')
    parser.add_argument('--multiple-levels', action='store_true', default=False,
                       help='生成多个增强级别的图像')
    parser.add_argument('--no-plate-recognition-mode', action='store_true',
                       help='禁用车牌识别优化模式，使用自定义增强设置')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日志级别 (默认: INFO)')
    
    # 增强配置选项
    parser.add_argument('--disable-noise-reduction', action='store_true',
                       help='禁用噪声减少')
    parser.add_argument('--disable-contrast-enhancement', action='store_true',
                       help='禁用对比度增强')
    parser.add_argument('--disable-sharpening', action='store_true',
                       help='禁用锐化处理')
    parser.add_argument('--disable-adaptive-thresholding', action='store_true',
                       help='禁用自适应阈值化')
    parser.add_argument('--disable-perspective-correction', action='store_true',
                       help='禁用透视校正')
    parser.add_argument('--disable-brightness-normalization', action='store_true',
                       help='禁用亮度标准化')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查源目录
    source_path = Path(args.source_dir)
    if not source_path.exists():
        logger.error(f"源目录不存在: {args.source_dir}")
        sys.exit(1)
    
    # 创建增强器配置
    if not args.no_plate_recognition_mode:
        # 车牌识别专用模式：只进行清晰化和降噪
        logger.info("使用车牌识别优化模式")
        config = {
            'enable_noise_reduction': True,
            'enable_contrast_enhancement': True,
            'enable_sharpening': True,
            'enable_adaptive_thresholding': False,  # 关闭二值化，保持原始颜色
            'enable_perspective_correction': False,  # 关闭透视校正
            'enable_brightness_normalization': True,
            'target_height': 64,
            'clahe_clip_limit': 1.5,  # 更温和的对比度增强，避免过亮图像变白
            'gaussian_kernel_size': 3,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
        }
    else:
        # 自定义模式：根据命令行参数设置
        logger.info("使用自定义增强模式")
        config = {
            'enable_noise_reduction': not args.disable_noise_reduction,
            'enable_contrast_enhancement': not args.disable_contrast_enhancement,
            'enable_sharpening': not args.disable_sharpening,
            'enable_adaptive_thresholding': not args.disable_adaptive_thresholding,
            'enable_perspective_correction': not args.disable_perspective_correction,
            'enable_brightness_normalization': not args.disable_brightness_normalization,
            'target_height': 64,
            'clahe_clip_limit': 3.0,
            'gaussian_kernel_size': 3,
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
        }
    
    # 创建增强器
    enhancer = PlateEnhancer(config)
    
    try:
        if args.multiple_levels:
            # 多级别增强
            logger.info("开始多级别批量增强处理...")
            logger.info(f"源目录: {args.source_dir}")
            logger.info(f"目标目录: {args.target_dir}")
            
            results = enhancer.batch_enhance_with_multiple_levels(
                args.source_dir,
                args.target_dir
            )
            
            # 输出结果统计
            logger.info("=== 多级别处理结果统计 ===")
            logger.info(f"总文件数: {results['total_count']}")
            logger.info(f"总成功数: {results['success_count']}")
            logger.info(f"总失败数: {results['error_count']}")
            
            for level, level_result in results['level_results'].items():
                logger.info(f"级别 {level}: 成功 {level_result['success_count']}/{level_result['total_count']}")
                
        else:
            # 单级别增强
            logger.info("开始单级别批量增强处理...")
            logger.info(f"源目录: {args.source_dir}")
            logger.info(f"目标目录: {args.target_dir}")
            logger.info(f"增强级别: {args.level}")
            
            results = enhancer.batch_enhance_directory(
                args.source_dir,
                args.target_dir,
                args.level
            )
            
            # 输出结果统计
            logger.info("=== 处理结果统计 ===")
            logger.info(f"总文件数: {results['total_count']}")
            logger.info(f"成功处理: {results['success_count']}")
            logger.info(f"处理失败: {results['error_count']}")
            
        if results['success_count'] > 0:
            logger.info("批量增强处理完成！")
        else:
            logger.warning("没有成功处理任何文件，请检查源目录和配置")
            
    except Exception as e:
        logger.error(f"批量处理过程中发生错误: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 