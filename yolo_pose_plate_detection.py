#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO-Pose车牌检测识别系统
包含检测、关键点定位、透视变换矫正和字符识别的完整流程
"""

import torch
import cv2
import numpy as np
import argparse
import copy
import time
import os
from pathlib import Path
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics import YOLO
from recongnition_moudle.plate_rec import get_plate_result, init_model, cv_imread
from recongnition_moudle.double_plate_split_merge import get_split_merge
from fonts.cv_puttext import cv2ImgAddText


def four_point_transform(image, pts):
    """
    透视变换得到车牌小图
    Args:
        image: 原始图像
        pts: 四个关键点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    Returns:
        warped: 透视变换后的车牌图像
    """
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect
    
    # 计算变换后的宽度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # 计算变换后的高度
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 目标四个点的坐标
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # 获取透视变换矩阵并进行变换
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def load_yolo_pose_model(weights_path, device):
    """
    加载YOLO-Pose模型
    Args:
        weights_path: 模型权重路径
        device: 设备 (cuda/cpu)
    Returns:
        model: 加载的模型
    """
    try:
        # 优先使用ultralytics的YOLO类加载pose模型
        model = YOLO(weights_path)
        model.to(device)
        return model
    except Exception as e:
        try:
            # 降级使用原始的attempt_load_weights（兼容您的原始代码）
            model = attempt_load_weights(weights_path, device=device)
            return model
        except Exception as e2:
            raise e2


def preprocess_image(image, img_size=640):
    """
    图像预处理
    Args:
        image: 输入图像
        img_size: 模型输入尺寸
    Returns:
        processed_img: 处理后的图像
        ratio: 缩放比例
        pad_w: 水平填充
        pad_h: 垂直填充
    """
    h, w = image.shape[:2]
    ratio = min(img_size / h, img_size / w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    
    # 调整图像大小
    resized_img = cv2.resize(image, (new_w, new_h))
    
    # 计算填充
    pad_w = (img_size - new_w) // 2
    pad_h = (img_size - new_h) // 2
    
    # 添加边框
    processed_img = cv2.copyMakeBorder(
        resized_img, pad_h, img_size - new_h - pad_h, 
        pad_w, img_size - new_w - pad_w, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    return processed_img, ratio, pad_w, pad_h


def postprocess_keypoints(keypoints, ratio, pad_w, pad_h):
    """
    后处理关键点坐标，转换回原图坐标系
    Args:
        keypoints: 检测到的关键点
        ratio: 缩放比例
        pad_w: 水平填充
        pad_h: 垂直填充
    Returns:
        processed_keypoints: 处理后的关键点
    """
    if keypoints is None or len(keypoints) == 0:
        return None
    
    # 去除填充并恢复到原图尺寸
    keypoints[:, 0] = (keypoints[:, 0] - pad_w) / ratio
    keypoints[:, 1] = (keypoints[:, 1] - pad_h) / ratio
    
    return keypoints


def detect_plate_with_pose(image, model, device, conf_threshold=0.5):
    """
    使用YOLO-Pose检测车牌和关键点
    Args:
        image: 输入图像
        model: YOLO-Pose模型
        device: 设备
        conf_threshold: 置信度阈值
    Returns:
        results: 检测结果列表，包含bbox和keypoints
    """
    results = []
    
    try:
        # 预处理图像
        processed_img, ratio, pad_w, pad_h = preprocess_image(image)
        
        # 进行推理
        with torch.no_grad():
            predictions = model(processed_img)
        
        # 处理ultralytics YOLO格式的结果
        if hasattr(predictions, '__iter__') and len(predictions) > 0:
            # 检查是否是ultralytics的Results对象
            pred_result = predictions[0] if isinstance(predictions, (list, tuple)) else predictions
            
            if hasattr(pred_result, 'boxes') and hasattr(pred_result, 'keypoints'):
                boxes = pred_result.boxes
                keypoints = pred_result.keypoints
                
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0].cpu().numpy())
                        if conf > conf_threshold:
                            # 获取边界框
                            bbox = box.xyxy[0].cpu().numpy()
                            
                            # 获取关键点
                            kpts = None
                            if keypoints is not None and i < len(keypoints):
                                kpts = keypoints[i].xy[0].cpu().numpy()  # [num_keypoints, 2]
                                # 后处理关键点坐标
                                kpts = postprocess_keypoints(kpts, ratio, pad_w, pad_h)
                            
                            # 后处理边界框坐标
                            bbox[[0, 2]] = (bbox[[0, 2]] - pad_w) / ratio
                            bbox[[1, 3]] = (bbox[[1, 3]] - pad_h) / ratio
                            
                            results.append({
                                'bbox': bbox.astype(int),
                                'keypoints': kpts,
                                'confidence': conf,
                                'class': int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else 0
                            })
                    return results
        
        # 处理传统的tensor输出格式
        if isinstance(predictions, (list, tuple)):
            pred = predictions[0]
        else:
            pred = predictions
            
        # 转换为numpy数组
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            
        # 检查预测结果的维度
        if len(pred.shape) == 3:  # [batch, num_detections, features]
            pred = pred[0]  # 取第一个batch
        elif len(pred.shape) == 2:  # [num_detections, features]
            pass  # 已经是正确的格式
        else:
            return results
            
        # 检查每个检测结果
        for i, detection in enumerate(pred):
            # 检查检测结果的长度
            if len(detection) < 5:
                continue
                
            # 安全地获取置信度
            try:
                conf = float(detection[4])
            except (IndexError, ValueError):
                continue
                
            if conf > conf_threshold:
                # 安全地解析边界框
                try:
                    cx, cy, w, h = detection[:4]
                    x1 = int((cx - w/2 - pad_w) / ratio)
                    y1 = int((cy - h/2 - pad_h) / ratio)
                    x2 = int((cx + w/2 - pad_w) / ratio)
                    y2 = int((cy + h/2 - pad_h) / ratio)
                except (IndexError, ValueError):
                    continue
                
                # 安全地获取类别
                cls = 0
                if len(detection) > 5:
                    try:
                        cls = int(detection[5])
                    except (IndexError, ValueError):
                        cls = 0
                
                # 安全地解析关键点
                kpts = None
                if len(detection) > 6:
                    try:
                        # 计算关键点数据的长度
                        kpts_data = detection[6:]
                        if len(kpts_data) >= 8:  # 至少4个点的坐标 (4*2=8)
                            # 重新整形为关键点坐标
                            kpts_data = kpts_data[:8].reshape(4, 2)  # 只取前4个关键点
                            kpts = postprocess_keypoints(kpts_data, ratio, pad_w, pad_h)
                    except Exception:
                        pass
                
                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'keypoints': kpts,
                    'confidence': conf,
                    'class': cls
                })
    
    except Exception as e:
        pass
    
    return results


def detect_plate_compatible(image, model, device, conf_threshold=0.5):
    """
    兼容原始 detect_rec_plate.py 代码的检测函数
    使用类似的处理流程
    """
    results = []
    
    try:
        # 使用原始代码的预处理方法
        from detect_rec_plate import pre_processing, post_processing
        
        # 创建一个简单的参数对象
        class Args:
            def __init__(self):
                self.img_size = 640
        
        opt = Args()
        
        # 预处理
        img_processed, r, left, top = pre_processing(image, opt, device)
        
        # 推理
        with torch.no_grad():
            predict = model(img_processed)[0]
        
        # 后处理
        outputs = post_processing(predict, conf_threshold, 0.5, r, left, top)
        
        # 转换为我们的格式
        for output in outputs:
            output_np = output.squeeze().cpu().numpy().tolist()
            
            if len(output_np) < 5:
                continue
                
            bbox = output_np[:4]
            bbox = [int(x) for x in bbox]
            
            confidence = output_np[4]
            cls = int(output_np[-1]) if len(output_np) > 5 else 0
            
            # 提取关键点
            kpts = None
            if len(output_np) >= 13:  # 5 + 8 (4个关键点 * 2坐标)
                landmarks = np.array(output_np[5:13], dtype='float32').reshape(4, 2)
                kpts = landmarks
            
            results.append({
                'bbox': bbox,
                'keypoints': kpts,
                'confidence': confidence,
                'class': cls
            })
            
    except Exception:
        # 如果兼容模式失败，回退到原始检测函数
        return detect_plate_with_pose(image, model, device, conf_threshold)
    
    return results


def extract_plate_roi(image, detection_result):
    """
    根据检测结果提取车牌ROI区域
    Args:
        image: 原始图像
        detection_result: 检测结果
    Returns:
        roi_image: 提取的车牌图像
        method: 提取方法 ('keypoints' 或 'bbox')
    """
    if detection_result['keypoints'] is not None and len(detection_result['keypoints']) >= 4:
        # 使用关键点进行透视变换
        try:
            keypoints = detection_result['keypoints'][:4]  # 取前4个关键点
            roi_image = four_point_transform(image, keypoints)
            return roi_image, 'keypoints'
        except Exception:
            pass
    
    # 使用边界框提取ROI
    bbox = detection_result['bbox']
    x1, y1, x2, y2 = bbox
    # 确保坐标在图像范围内
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi_image = image[y1:y2, x1:x2]
    return roi_image, 'bbox'


def recognize_plate(roi_image, plate_rec_model, device, is_double_plate=False):
    """
    识别车牌字符
    Args:
        roi_image: 车牌ROI图像
        plate_rec_model: 车牌识别模型
        device: 设备
        is_double_plate: 是否为双层车牌
    Returns:
        plate_result: 识别结果字典
    """
    try:
        if roi_image is None or roi_image.size == 0:
            return None
        
        # 如果是双层车牌，进行分割合并处理
        if is_double_plate:
            roi_image = get_split_merge(roi_image)
            
        is_color = False
        color_conf = 0
        plate_color = ""
        # 进行车牌字符识别
        if is_color:
            plate_number, rec_prob, plate_color, color_conf = get_plate_result(
                roi_image, device, plate_rec_model, is_color=True
            )
        else:
            plate_number, rec_prob = get_plate_result(
                roi_image, device, plate_rec_model, is_color=False
            )
        
        result = {
            'plate_number': plate_number,
            'plate_color': plate_color,
            'recognition_confidence': rec_prob,
            'color_confidence': color_conf,
            'roi_height': roi_image.shape[0],
            'roi_width': roi_image.shape[1],
            'is_double_plate': is_double_plate
        }
        
        return result
    
    except Exception:
        return None


def draw_results(image, detections, recognition_results):
    """
    在图像上绘制检测和识别结果（正确显示中文）
    Args:
        image: 原始图像
        detections: 检测结果列表
        recognition_results: 识别结果列表
    Returns:
        annotated_image: 标注后的图像
    """
    annotated_image = image.copy()
    
    for i, (detection, recognition) in enumerate(zip(detections, recognition_results)):
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # 绘制边界框
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制关键点
        if detection['keypoints'] is not None:
            keypoints = detection['keypoints']
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
            for j, (kx, ky) in enumerate(keypoints[:4]):
                if 0 <= kx < image.shape[1] and 0 <= ky < image.shape[0]:
                    cv2.circle(annotated_image, (int(kx), int(ky)), 5, colors[j % 4], -1)
        
        # 绘制识别结果（使用中文显示）
        if recognition is not None:
            plate_text = f"{recognition['plate_number']} {recognition['plate_color']}"
            if recognition['is_double_plate']:
                plate_text += " 双层"
            
            # 使用cv2ImgAddText正确显示中文
            try:
                # 绘制白色背景
                labelSize = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                if x1 + labelSize[0] > image.shape[1]:
                    x1 = int(image.shape[1] - labelSize[0])
                cv2.rectangle(annotated_image, 
                             (x1, int(y1 - round(1.6 * labelSize[1]))), 
                             (int(x1 + round(1.2 * labelSize[0])), y1), 
                             (255, 255, 255), cv2.FILLED)
                
                # 使用cv2ImgAddText绘制中文文本
                annotated_image = cv2ImgAddText(annotated_image, plate_text, x1, 
                                              int(y1 - round(1.6 * labelSize[1])), 
                                              (0, 0, 0), 21)
            except Exception:
                # 如果cv2ImgAddText失败，使用英文替代
                conf_text = f"Plate {i+1}: {detection['confidence']:.2f}"
                cv2.putText(annotated_image, conf_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_image


def process_single_image(image_path, pose_model, rec_model, device, output_dir=None):
    """
    处理单张图像
    Args:
        image_path: 图像路径
        pose_model: YOLO-Pose模型
        rec_model: 车牌识别模型
        device: 设备
        output_dir: 输出目录
    Returns:
        results: 处理结果
    """
    print(f"处理图像: {image_path}")
    
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    
    start_time = time.time()
    
    # 1. 使用YOLO-Pose检测车牌和关键点
    # 首先尝试兼容模式
    try:
        detections = detect_plate_compatible(image, pose_model, device)
    except Exception:
        detections = detect_plate_with_pose(image, pose_model, device)
    
    print(f"检测到 {len(detections)} 个车牌")
    
    # 2. 对每个检测结果进行处理
    recognition_results = []
    roi_images = []
    
    for i, detection in enumerate(detections):
        print(f"处理第 {i+1} 个车牌，置信度: {detection['confidence']:.3f}")
        
        # 提取车牌ROI
        roi_image, method = extract_plate_roi(image, detection)
        roi_images.append(roi_image)
        
        # 判断是否为双层车牌（基于检测类别）
        is_double_plate = detection['class'] == 1  # 假设类别1为双层车牌
        
        # 识别车牌
        recognition_result = recognize_plate(roi_image, rec_model, device, is_double_plate)
        recognition_results.append(recognition_result)
        
        if recognition_result:
            print(f"识别结果: {recognition_result['plate_number']} {recognition_result['plate_color']}")
        else:
            print("识别失败")
    
    # 3. 绘制结果
    annotated_image = draw_results(image, detections, recognition_results)
    
    # 4. 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存标注图像
        image_name = Path(image_path).stem
        output_path = output_dir / f"{image_name}_result.jpg"
        cv2.imwrite(str(output_path), annotated_image)
        
        # 保存ROI图像
        for i, roi_img in enumerate(roi_images):
            if roi_img is not None and roi_img.size > 0:
                roi_path = output_dir / f"{image_name}_roi_{i}.jpg"
                cv2.imwrite(str(roi_path), roi_img)
    
    processing_time = time.time() - start_time
    print(f"处理耗时: {processing_time:.3f}秒\n")
    
    return {
        'detections': detections,
        'recognition_results': recognition_results,
        'annotated_image': annotated_image,
        'roi_images': roi_images,
        'processing_time': processing_time
    }


def main():
    parser = argparse.ArgumentParser(description='YOLO-Pose车牌检测识别系统')
    parser.add_argument('--pose_model', type=str, default='weights/plate_pose.pt',
                       help='YOLO-Pose模型路径')
    parser.add_argument('--rec_model', type=str, default='weights/checkpoint_85_acc_0.9509.pth',
                       help='车牌识别模型路径')
    parser.add_argument('--source', type=str, default=r'D:\Data\car_plate\plate_recong\live\ID1928278804434837504\ID1928278804434837504\images',
                       help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default=r'D:\Data\car_plate\plate_recong\live\ID1928278804434837504\ID1928278804434837504\images_result',
                       help='输出目录')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                       help='检测置信度阈值')
    
    args = parser.parse_args()
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("加载YOLO-Pose模型...")
    pose_model = load_yolo_pose_model(args.pose_model, device)
    
    print("加载车牌识别模型...")
    rec_model = init_model(device, args.rec_model, is_color=False)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # 处理输入
    source_path = Path(args.source)
    
    if source_path.is_file():
        # 处理单个文件
        process_single_image(source_path, pose_model, rec_model, device, args.output)
    elif source_path.is_dir():
        # 处理目录中的所有图像
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG'}
        image_files = [f for f in source_path.iterdir() 
                      if f.suffix in image_extensions]
        
        total_time = 0
        successful_count = 0
        
        print(f"在目录 {source_path} 中找到 {len(image_files)} 个图像文件")
        
        for image_file in image_files:
            result = process_single_image(image_file, pose_model, rec_model, device, args.output)
            if result:
                total_time += result['processing_time']
                successful_count += 1
        
        if successful_count > 0:
            avg_time = total_time / successful_count
            print(f"处理完成！总共处理 {successful_count} 张图像，平均耗时: {avg_time:.3f}秒")
    else:
        print(f"输入路径不存在: {args.source}")


if __name__ == "__main__":
    main() 