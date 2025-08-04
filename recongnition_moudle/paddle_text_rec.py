from paddleocr import PaddleOCR
import cv2
import os
import time
from paddleocr import TextRecognition

class AccuracyCalculator:
    """
    准确率计算器
    """
    def __init__(self):
        self.total_count = 0
        self.exact_match_count = 0
        self.char_total_count = 0
        self.char_correct_count = 0
        self.results_detail = []
    
    def extract_label_from_filename(self, filename):
        """
        从文件名提取车牌号标签
        例如：鄂FT0960_1473.jpg -> 鄂FT0960
        """
        basename = os.path.splitext(filename)[0]  # 去掉扩展名
        if '_' in basename:
            label = basename.split('_')[0]  # 取第一部分作为标签
        else:
            label = basename  # 如果没有下划线，整个文件名就是标签
        return label
    
    def calculate_char_accuracy(self, predicted, ground_truth):
        """
        计算字符级准确率
        """
        if not ground_truth:
            return 0.0, 0, 0
        
        # 转换为字符列表进行比较
        pred_chars = list(predicted) if predicted else []
        gt_chars = list(ground_truth)
        
        correct_chars = 0
        total_chars = len(gt_chars)
        
        # 逐字符比较
        min_len = min(len(pred_chars), len(gt_chars))
        for i in range(min_len):
            if pred_chars[i] == gt_chars[i]:
                correct_chars += 1
        
        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
        return char_accuracy, correct_chars, total_chars
    
    def add_result(self, filename, predicted_text, confidence):
        """
        添加识别结果并计算准确率
        """
        ground_truth = self.extract_label_from_filename(filename)
        
        # 清理预测文本（移除可能的空格等）
        predicted_clean = predicted_text.strip() if predicted_text else ""
        ground_truth_clean = ground_truth.strip()
        
        # 完全匹配准确率
        is_exact_match = predicted_clean == ground_truth_clean
        if is_exact_match:
            self.exact_match_count += 1
        
        # 字符级准确率
        char_acc, char_correct, char_total = self.calculate_char_accuracy(predicted_clean, ground_truth_clean)
        self.char_correct_count += char_correct
        self.char_total_count += char_total
        
        self.total_count += 1
        
        # 保存详细结果
        result_detail = {
            'filename': filename,
            'ground_truth': ground_truth_clean,
            'predicted': predicted_clean,
            'confidence': confidence,
            'exact_match': is_exact_match,
            'char_accuracy': char_acc
        }
        self.results_detail.append(result_detail)
        
        return result_detail
    
    def get_statistics(self):
        """
        获取统计信息
        """
        if self.total_count == 0:
            return {
                'total_images': 0,
                'exact_match_accuracy': 0.0,
                'character_accuracy': 0.0,
                'exact_match_count': 0,
                'char_correct_count': 0,
                'char_total_count': 0
            }
        
        exact_match_accuracy = self.exact_match_count / self.total_count
        character_accuracy = self.char_correct_count / self.char_total_count if self.char_total_count > 0 else 0.0
        
        return {
            'total_images': self.total_count,
            'exact_match_accuracy': exact_match_accuracy,
            'character_accuracy': character_accuracy,
            'exact_match_count': self.exact_match_count,
            'char_correct_count': self.char_correct_count,
            'char_total_count': self.char_total_count
        }
    
    def print_statistics(self):
        """
        打印统计信息
        """
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("识别准确率统计报告")
        print("="*50)
        print(f"总图片数量: {stats['total_images']}")
        print(f"完全匹配数量: {stats['exact_match_count']}")
        print(f"完全匹配准确率: {stats['exact_match_accuracy']:.2%}")
        print(f"字符级准确率: {stats['character_accuracy']:.2%}")
        print(f"字符正确数/总数: {stats['char_correct_count']}/{stats['char_total_count']}")
        print("="*50)
    
    def print_error_cases(self, max_show=10):
        """
        打印识别错误的案例
        """
        error_cases = [result for result in self.results_detail if not result['exact_match']]
        
        if not error_cases:
            print("\n没有识别错误的案例！")
            return
        
        print(f"\n识别错误案例 (显示前{min(max_show, len(error_cases))}个):")
        print("-" * 80)
        for i, case in enumerate(error_cases[:max_show]):
            print(f"{i+1:2d}. 文件: {case['filename']}")
            print(f"    标签: {case['ground_truth']}")
            print(f"    预测: {case['predicted']}")
            print(f"    置信度: {case['confidence']:.3f}")
            print(f"    字符准确率: {case['char_accuracy']:.2%}")
            print()

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

        # 初始化准确率计算器
        self.accuracy_calculator = AccuracyCalculator()
        self.model_rec = TextRecognition(model_dir=model_path)

    def recognize(self,image_path):
        # ppocr_result = self.ppocr.predict(input=image_path)
        ppocr_result = self.model_rec.predict(input=image_path)
        
        # 提取有用信息
        if ppocr_result and len(ppocr_result) > 0:
            ocr_result = ppocr_result[0]  # 取第一个结果
            
            # 提取文字、置信度和坐标信息
            text = ocr_result.get('rec_text', [])
            score = ocr_result.get('rec_score', [])
            # boxes = ocr_result.get('rec_boxes', [])
            
            # 组织返回结果
            result = {
                'text': text if text else '',
                'confidence': score if score else 0.0,
                # 'box': boxes[i].tolist() if i < len(boxes) else []
            }

            
            return result
        else:
            return []

    def single_image_recognize(self,image_path):
        #image = cv2.imread(image_path)
        result = self.recognize(image_path)
        return result
    
    def batch_recognize(self,image_dir,output_dir, enable_accuracy_stats=True):
        #results = []
        ocr_result_num = 0
        no_result_num = 0
        os.makedirs(output_dir,exist_ok=True)
        
        # 重置准确率计算器
        if enable_accuracy_stats:
            self.accuracy_calculator = AccuracyCalculator()
        
        image_paths = [os.path.join(image_dir,img) for img in os.listdir(image_dir)]
        
        print(f"开始批量识别，共{len(image_paths)}张图片...")
        
        for image_path in image_paths:
            image = cv2.imread(image_path)
            if image is None:
                print(f"警告: 无法读取图片 {image_path}")
                continue
                
            filename = os.path.basename(image_path)
            t1 = time.time()
            result = self.single_image_recognize(image_path)
            t2 = time.time()
            
            print(f"识别时间：{t2-t1:.3f}秒 - {filename}")
            
            if result:
                ocr_result_num += 1
                predicted_text = result['text']
                confidence = result['confidence']
                
                # 添加到准确率统计
                if enable_accuracy_stats:
                    accuracy_detail = self.accuracy_calculator.add_result(filename, predicted_text, confidence)
                    
                image_name = filename.split(".")[0]
                new_image_name = f"{predicted_text}_{ocr_result_num}.jpg"
                print(f"已识别：{new_image_name}，识别结果：{predicted_text} (置信度: {confidence:.3f})")
                
                # 如果启用准确率统计，显示当前结果的准确性
                if enable_accuracy_stats:
                    match_status = "✓" if accuracy_detail['exact_match'] else "✗"
                    print(f"  {match_status} 标签: {accuracy_detail['ground_truth']} | 字符准确率: {accuracy_detail['char_accuracy']:.2%}")
                
                # cv2.imwrite(os.path.join(output_dir,new_image_name),image)
            else:
                no_result_num += 1
                print(f"未识别到文字：{filename}")
                
                # 对于未识别的图片，也要添加到统计中
                if enable_accuracy_stats:
                    self.accuracy_calculator.add_result(filename, "", 0.0)
            
        print(f"\n识别完成！")
        print(f"识别总数：{ocr_result_num}")
        print(f"未识别总数：{no_result_num}")
        
        # 打印准确率统计
        if enable_accuracy_stats:
            self.accuracy_calculator.print_statistics()
            self.accuracy_calculator.print_error_cases(max_show=10)
        
        return self.accuracy_calculator.get_statistics() if enable_accuracy_stats else None

    def get_accuracy_stats(self):
        """
        获取当前的准确率统计信息
        """
        return self.accuracy_calculator.get_statistics()

if __name__ == "__main__":
    model_path = "./weights/PP-OCRv5_server_rec_infer4"
    ppocr = PaddleOCRRec(model_path)
    img_path = "./data/imgs/2.jpg"
    img_dir = "/expdata/givap/data/plate_recong/mix/mix_b1"
    output_dir = "./data/output2_ppocr_mix_b1"
    
    # 单张图片测试
    #result = ppocr.single_image_recognize(img_path)
    #print(result)
    
    # 批量识别并统计准确率
    stats = ppocr.batch_recognize(img_dir, output_dir, enable_accuracy_stats=True)
    
    # 也可以单独获取统计信息
    #final_stats = ppocr.get_accuracy_stats()
    #print(f"\n最终统计: {final_stats}")

