import torch
import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
from ultralytics.nn.tasks import attempt_load_weights
from ultralytics import YOLO
from paddleocr import TextDetection
from recongnition_moudle.plate_rec import init_model, get_plate_result, cv_imread
from fonts.cv_puttext import cv2ImgAddText
import os

class YoloPaddleDetctorCRNN:
    """YOLO + paddle + crnn 车牌检测"""

    def __init__(self, yolo_model_path, crnn_model_path):


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO(yolo_model_path)
        self.text_detection = TextDetection(model_name="PP-OCRv5_server_det")
        self.crnn_model = init_model(self.device,crnn_model_path)

    def _load_yolo_model(self,model_path):
        try:
            self.yolo_model = YOLO(model_path)
            self.yolo_model.to(self.device)
            
            return self.yolo_model
        except Exception as e:
            raise ValueError(f"Error loading YOLO model: {e}")
        



    def _detect_plate(self, image,conf_thresh=0.25):
        """
        YOLO 车牌检测
        Args:
            image: 输入图像
            conf_thresh: 置信度阈值
        Returns:
            results: 检测结果
        """
        results = []
        try:
            prediction = self.yolo_model(image)

            boxes = prediction[0].boxes
            
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                if conf < conf_thresh:
                    continue
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                cls = int(box.cls[0].item())
                class_name = self.yolo_model.names[cls]

                results.append({
                    'bbox':bbox,
                    'class_name':class_name,
                    'class_id':cls
                })
        except Exception as e:
            print(f"YOLO 车牌检测失败：{e}")

        return results
    
    def _detect_text(self,roi_img):
        """
        OCR 车牌文字检测
        Args:
            roi_img: 车牌区域图像
        Returns:
            text_boxes: 车牌文字检测框坐标列表
        """
        text_boxes = []
        try:
            # 使用PaddleOCR文本检测模块进行文本区域检测
            text_result = self.text_detection.predict(roi_img,batch_size=1)
            
            # 提取检测结果
            dt_polys = text_result[0].get('dt_polys', None)

            dt_scores = text_result[0].get('dt_scores', [])
            
            # 遍历每个检测到的多边形
            for i, poly in enumerate(dt_polys):
                if len(poly) >= 4:
                    # poly是numpy数组，包含多个[x, y]坐标点
                    poly_array = np.array(poly)
                    
                    # 计算边界框的最小外接矩形
                    x_min = int(np.min(poly_array[:, 0]))
                    y_min = int(np.min(poly_array[:, 1]))
                    x_max = int(np.max(poly_array[:, 0]))
                    y_max = int(np.max(poly_array[:, 1]))
                    
                    # 获取对应的置信度分数
                    confidence = dt_scores[i] if i < len(dt_scores) else 0.9
                    
                    # 存储文本框信息
                    text_box = {
                        'bbox': [x_min, y_min, x_max, y_max],  # 边界框格式 [x1, y1, x2, y2]
                        'poly': poly.tolist(),  # 转换numpy数组为列表
                        'confidence': float(confidence)  # 置信度分数
                    }
                    text_boxes.append(text_box)
                        

                        
        except Exception as e:
            print(f"OCR 车牌文字检测失败：{e}")
            return []
        
        return text_boxes
    
    def _crnn_recognize(self,roi_img):
        """
        CRNN 车牌文字识别
        Args:
            roi_img: 车牌区域图像
        Returns:
            text_result: 车牌文字识别结果
        """

        try:

            result,prob = get_plate_result(roi_img,self.device,self.crnn_model)
            return result
        except Exception as e:
            print(f"CRNN 车牌文字识别失败：{e}")
            return ""
    def plate_recognize(self,image):
        """
        车牌识别
        Args:
            image: 输入图像
        Returns:
            plate_result: 车牌识别结果  
        """

        plate_recong_results = []
        try:
            plate_results = self._detect_plate(image)
            for plate_result in plate_results:
                bbox = plate_result['bbox']
                roi_img = self._get_roi_img(image,bbox)
                # x1,y1,x2,y2 = bbox
                # roi_img = image[y1:y2,x1:x2]
                text_boxes = self._detect_text(roi_img)
                if len(text_boxes) == 1: # 只检测到一段文字
                    text_box = text_boxes[0]
                    text_roi = self._get_roi_img(image,text_box['bbox'])
                    text = self._crnn_recognize(text_roi)
                    plate_recong_results.append({
                        'text':text,
                        'bbox':bbox,
                    })
                elif len(text_boxes) > 1: # 双层车牌的情况
                    text_all = ""
                    for text_box in text_boxes:
                        text_roi = self._get_roi_img(image,text_box['bbox'])
                        text = self._crnn_recognize(text_roi)
                        text_all += text
                        plate_result.append({
                            'text':text,
                            'text_box':text_box,
                        })
                else:
                    print("未检测到车牌文字")
        except Exception as e:
            print(f"车牌识别失败：{e}")
        return plate_recong_results
    def _get_roi_img(self,image,bbox):
        """
            截取图像
            Args:
                image: 输入图像
                bbox: 车牌区域坐标
            Returns:
                roi_img: 截取的图像
        """
        x1,y1,x2,y2 = bbox
        roi_img = image[y1:y2,x1:x2]
        return roi_img
    def _draw_plate_result(self,image,plate_results):
        """
        绘制车牌识别结果
        Args:
            image: 输入图像
            plate_result: 车牌识别结果
        """
        for plate_result in plate_results:
            bbox = plate_result['bbox']
            x1,y1,x2,y2 = bbox
            text = plate_result['text']
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2) #画框
            labelSize = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
            if x1+labelSize[0][0]>image.shape[1]:                 #防止显示的文字越界
                x1=int(image.shape[1]-labelSize[0][0])
            image = cv2ImgAddText(image,text,x1,int(y1-round(1.6*labelSize[0][1])),(0,0,0),21) #添加文字
        

        return image
    
    def batch_plate_recognize(self,img_dir,output_dir):
        """
        批量车牌识别
        Args:
            images: 输入图像列表
        Returns:
            plate_results: 车牌识别结果列表
        """
        img_list = os.listdir(img_dir)
        for img_name in img_list:
            img_path = os.path.join(img_dir,img_name)
            image = cv2.imread(img_path)
            plate_results = self.plate_recognize(image)
            image = self._draw_plate_result(image,plate_results)
            cv2.imwrite(os.path.join(output_dir,img_name),image)
            print(f"车牌识别结果已保存到{output_dir}")
            # cv2.imshow("plate_result",image)
            # cv2.waitKey(0)






if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--img_dir",type=str,default="data/imgs")
    args.add_argument("--output_dir",type=str,default="data/output")
    args.add_argument("--yolo_model_path",type=str,default="./weights/yolo_det.pt")
    args.add_argument("--crnn_model_path",type=str,default="./weights/plate_rec_color.pth")
    # os.makedirs(args.output_dir,exist_ok=True)
        
    args = args.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    yolo_paddle_detect = YoloPaddleDetctorCRNN(yolo_model_path=args.yolo_model_path,crnn_model_path=args.crnn_model_path)
    if os.path.isdir(args.img_dir):
        yolo_paddle_detect.batch_plate_recognize(args.img_dir,args.output_dir)
    else:
        image = cv2.imread(args.img_dir)
        plate_results = yolo_paddle_detect.plate_recognize(image)
        image = yolo_paddle_detect._draw_plate_result(image,plate_results)
        print(f"{args.img_dir}车牌识别结果：{plate_results}")
        cv2.imwrite(os.path.join(args.output_dir,os.path.basename(args.img_dir)),image)
        print(f"车牌识别结果已保存到{args.output_dir}")
        cv2.imshow("plate_result",image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()