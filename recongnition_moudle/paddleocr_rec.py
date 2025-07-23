from paddleocr import PaddleOCR
import cv2
import os
import time


class PaddleOCRRec:
    """
    使用PaddleOCR进行车牌识别
    Args:
        model_path: 模型路径
    Returns:
        result: 识别结果
    """
    # 单例模式
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PaddleOCRRec, cls).__new__(cls)
        return cls._instance

    def __init__(self,model_path):
        self.ppocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)

    def recognize(self,image_path):
        ppocr_result = self.ppocr.predict(input=image_path)
        
        # 提取有用信息
        if ppocr_result and len(ppocr_result) > 0:
            ocr_result = ppocr_result[0]  # 取第一个结果
            
            # 提取文字、置信度和坐标信息
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
            boxes = ocr_result.get('rec_boxes', [])
            
            # 组织返回结果
            results = []
            for i in range(len(texts)):
                item = {
                    'text': texts[i] if i < len(texts) else '',
                    'confidence': scores[i] if i < len(scores) else 0.0,
                    'box': boxes[i].tolist() if i < len(boxes) else []
                }
                results.append(item)
            
            return results
        else:
            return []
    def single_image_recognize(self,image_path):
        #image = cv2.imread(image_path)
        result = self.recognize(image_path)
        return result
    
    def batch_recognize(self,image_dir,output_dir):
        #results = []
        ocr_result_num = 0
        no_result_num = 0
        os.makedirs(output_dir,exist_ok=True)
        image_paths = [os.path.join(image_dir,img) for img in os.listdir(image_dir)]
        for image_path in image_paths:
            image = cv2.imread(image_path)
            t1 = time.time()
            result = self.single_image_recognize(image_path)
            t2 = time.time()
            print(f"识别时间：{t2-t1}秒")
            if result:
                ocr_result_num += 1
                image_name = os.path.basename(image_path)
                image_name = image_name.split(".")[0]
                # text = result[0]['text']
                # if text[0] != "T":
                #     if len(text) == 3:
                #         text = "T" + text
                #     elif len(text) == 4:
                #         text[0] = "T" 

                image_name = f"{result[0]['text']}_{ocr_result_num}.jpg"
                print(f"已识别：{image_name}，识别结果：{result[0]['text']}")
                cv2.imwrite(os.path.join(output_dir,image_name),image)
            #print(result)
            #results.append(result)
        print(f"识别总数：{ocr_result_num}")
        print(f"未识别总数：{no_result_num}")

        
    
if __name__ == "__main__":
    model_path = "./weights/PP-OCRv5_server_rec"
    ppocr = PaddleOCRRec(model_path)
    img_path = "./data/imgs/2.jpg"
    img_dir = "/expdata/givap/dataset/test_data/unstand_plate/b1"
    output_dir = "./data/output_ppocr1"
    #result = ppocr.single_image_recognize(img_path)
    #print(result)
    #image_paths = ["./data/imgs/1.jpg","./data/imgs/2.jpg","./data/imgs/3.jpg"]
    ppocr.batch_recognize(img_dir,output_dir)

