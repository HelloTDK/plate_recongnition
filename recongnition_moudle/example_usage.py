"""
PaddleOCR识别模块使用示例
展示单图识别、批量识别和结果可视化的用法
"""

import os
import json
from paddleocr_rec import PaddleOCRRec, recognize_image, recognize_folder


def example_single_image_recognition():
    """单图识别示例"""
    print("=== 单图识别示例 ===")
    
    # 初始化识别器
    ocr = PaddleOCRRec()
    
    # 图片路径（请替换为实际的图片路径）
    image_path = "./test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图片不存在: {image_path}")
        print("请将图片路径替换为实际存在的图片")
        return
    
    # 识别图片，设置最小置信度为0.5
    result = ocr.recognize_single_image(image_path, min_confidence=0.5)
    
    print(f"识别状态: {result['status']}")
    if result['status'] == 'success':
        print(f"图片路径: {result['input_path']}")
        print(f"检测到的文本数量: {result['total_detections']}")
        print(f"平均置信度: {result['avg_confidence']:.4f}")
        print(f"所有文本: {result['all_text']}")
        
        print("\n带置信度的文本列表:")
        for text_info in result['texts_with_confidence']:
            print(f"  文本: '{text_info['text']}', 置信度: {text_info['confidence']:.4f}")
    else:
        print(f"识别失败: {result['error_message']}")


def example_batch_recognition():
    """批量识别示例"""
    print("\n=== 批量识别示例 ===")
    
    # 图片文件夹路径（请替换为实际的文件夹路径）
    folder_path = "./test_images"
    
    if not os.path.exists(folder_path):
        print(f"示例文件夹不存在: {folder_path}")
        print("请将文件夹路径替换为实际存在的图片文件夹")
        return
    
    # 批量识别
    batch_result = recognize_folder(
        folder_path=folder_path,
        min_confidence=0.5,
        save_results=True
    )
    
    print(f"批量识别状态: {batch_result['status']}")
    print(f"文件夹路径: {batch_result['folder_path']}")
    print(f"总文件数: {batch_result['total_files']}")
    print(f"成功识别: {batch_result['successful_count']}")
    print(f"识别失败: {batch_result['failed_count']}")
    
    if batch_result['successful_count'] > 0:
        summary = batch_result['summary']
        print(f"平均置信度: {summary['avg_confidence']:.4f}")
        print(f"总检测数: {summary['total_detections']}")
        
        print("\n各文件识别结果:")
        for result in batch_result['results']:
            if result['status'] == 'success':
                print(f"  {result['filename']}: {len(result['texts_with_confidence'])}个文本")
            else:
                print(f"  {result['filename']}: 识别失败")


def example_structured_data_usage():
    """结构化数据使用示例"""
    print("\n=== 结构化数据使用示例 ===")
    
    # 使用原始recognize方法获取OCRResult对象
    ocr = PaddleOCRRec()
    image_path = "./test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"示例图片不存在: {image_path}")
        return
    
    # 获取结构化结果
    ocr_result = ocr.recognize(image_path)
    
    print("原始数据属性:")
    print(f"  输入路径: {ocr_result.input_path}")
    print(f"  角度: {ocr_result.angle}")
    print(f"  文本类型: {ocr_result.text_type}")
    print(f"  识别文本数量: {len(ocr_result.recognition_texts)}")
    
    # 获取高置信度文本
    high_confidence_texts = ocr_result.get_text_with_confidence(min_confidence=0.8)
    print(f"\n高置信度文本(>0.8): {len(high_confidence_texts)}个")
    for text_info in high_confidence_texts:
        print(f"  {text_info['text']} (置信度: {text_info['confidence']:.4f})")
    
    # 转换为字典用于可视化
    dict_result = ocr_result.to_dict()
    print(f"\n转换为字典格式，包含{len(dict_result)}个字段")


def example_visualization_data_preparation():
    """为可视化准备数据的示例"""
    print("\n=== 可视化数据准备示例 ===")
    
    # 模拟批量识别结果
    ocr = PaddleOCRRec()
    
    # 这里可以添加实际的可视化数据准备逻辑
    # 比如为前端图表准备数据
    
    visualization_data = {
        'charts': {
            'confidence_distribution': {
                'title': '置信度分布',
                'type': 'histogram',
                'data': []  # 实际数据需要从识别结果中提取
            },
            'text_length_distribution': {
                'title': '文本长度分布',
                'type': 'bar',
                'data': []
            },
            'detection_count_per_image': {
                'title': '每张图片检测数量',
                'type': 'line',
                'data': []
            }
        },
        'summary_stats': {
            'total_images': 0,
            'total_text_detections': 0,
            'avg_confidence': 0.0,
            'success_rate': 0.0
        }
    }
    
    print("可视化数据结构已准备完成")
    print(json.dumps(visualization_data, ensure_ascii=False, indent=2))


def main():
    """主函数，运行所有示例"""
    print("PaddleOCR识别模块使用示例\n")
    
    try:
        example_single_image_recognition()
        example_batch_recognition()
        example_structured_data_usage()
        example_visualization_data_preparation()
        
    except Exception as e:
        print(f"运行示例时出错: {e}")
        print("请确保已安装PaddleOCR并且图片路径正确")


if __name__ == "__main__":
    main() 