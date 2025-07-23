"""
可视化数据处理工具
为OCR识别结果提供可视化数据处理功能
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


class OCRVisualizationProcessor:
    """OCR识别结果可视化数据处理器"""
    
    def __init__(self):
        """初始化可视化处理器"""
        self.results = []
        self.stats = {}
    
    def load_results(self, results: List[Dict]):
        """
        加载识别结果数据
        
        Args:
            results: 识别结果列表
        """
        self.results = [r for r in results if r.get('status') == 'success']
        self._calculate_stats()
    
    def load_from_json(self, json_file: str):
        """
        从JSON文件加载批量识别结果
        
        Args:
            json_file: JSON结果文件路径
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'results' in data:
            self.load_results(data['results'])
        else:
            self.load_results([data])  # 单个结果
    
    def _calculate_stats(self):
        """计算统计信息"""
        if not self.results:
            self.stats = {}
            return
        
        confidences = []
        text_lengths = []
        detection_counts = []
        all_texts = []
        
        for result in self.results:
            # 置信度
            if 'texts_with_confidence' in result:
                for text_info in result['texts_with_confidence']:
                    confidences.append(text_info['confidence'])
                    text_lengths.append(len(text_info['text']))
                    all_texts.append(text_info['text'])
            
            # 每张图片的检测数量
            detection_counts.append(result.get('total_detections', 0))
        
        self.stats = {
            'total_images': len(self.results),
            'total_detections': sum(detection_counts),
            'confidences': confidences,
            'text_lengths': text_lengths,
            'detection_counts': detection_counts,
            'all_texts': all_texts,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'avg_detection_per_image': np.mean(detection_counts) if detection_counts else 0
        }
    
    def get_confidence_distribution(self, bins: int = 10) -> Dict:
        """
        获取置信度分布数据
        
        Args:
            bins: 直方图分组数量
            
        Returns:
            Dict: 置信度分布数据
        """
        if not self.stats.get('confidences'):
            return {'bins': [], 'counts': [], 'bin_edges': []}
        
        confidences = self.stats['confidences']
        counts, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))
        
        # 计算每个区间的中心点
        bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]
        
        return {
            'title': '置信度分布',
            'type': 'histogram',
            'bins': bin_centers,
            'counts': counts.tolist(),
            'bin_edges': bin_edges.tolist(),
            'total_count': len(confidences),
            'avg_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences))
        }
    
    def get_text_length_distribution(self, max_length: int = 50) -> Dict:
        """
        获取文本长度分布数据
        
        Args:
            max_length: 最大显示长度
            
        Returns:
            Dict: 文本长度分布数据
        """
        if not self.stats.get('text_lengths'):
            return {'lengths': [], 'counts': []}
        
        text_lengths = [min(length, max_length) for length in self.stats['text_lengths']]
        length_counter = Counter(text_lengths)
        
        lengths = sorted(length_counter.keys())
        counts = [length_counter[length] for length in lengths]
        
        return {
            'title': '文本长度分布',
            'type': 'bar',
            'lengths': lengths,
            'counts': counts,
            'avg_length': float(np.mean(self.stats['text_lengths'])),
            'max_length_shown': max_length
        }
    
    def get_detection_count_analysis(self) -> Dict:
        """
        获取检测数量分析数据
        
        Returns:
            Dict: 检测数量分析数据
        """
        if not self.stats.get('detection_counts'):
            return {'image_indices': [], 'detection_counts': []}
        
        detection_counts = self.stats['detection_counts']
        image_indices = list(range(1, len(detection_counts) + 1))
        
        return {
            'title': '每张图片检测数量',
            'type': 'line',
            'image_indices': image_indices,
            'detection_counts': detection_counts,
            'avg_detection': float(np.mean(detection_counts)),
            'max_detection': int(np.max(detection_counts)),
            'min_detection': int(np.min(detection_counts))
        }
    
    def get_text_frequency_analysis(self, top_n: int = 20) -> Dict:
        """
        获取文本频率分析
        
        Args:
            top_n: 显示前N个高频词
            
        Returns:
            Dict: 文本频率分析数据
        """
        if not self.stats.get('all_texts'):
            return {'words': [], 'frequencies': []}
        
        # 简单的词频统计（可以根据需要改进分词逻辑）
        all_text = ' '.join(self.stats['all_texts'])
        words = all_text.split()
        word_counter = Counter(words)
        
        # 获取前N个高频词
        top_words = word_counter.most_common(top_n)
        words, frequencies = zip(*top_words) if top_words else ([], [])
        
        return {
            'title': f'前{top_n}高频词',
            'type': 'bar',
            'words': list(words),
            'frequencies': list(frequencies),
            'total_words': len(words),
            'unique_words': len(word_counter)
        }
    
    def get_confidence_vs_length_scatter(self) -> Dict:
        """
        获取置信度与文本长度的散点图数据
        
        Returns:
            Dict: 散点图数据
        """
        if not self.stats.get('confidences') or not self.stats.get('text_lengths'):
            return {'confidences': [], 'text_lengths': []}
        
        confidences = self.stats['confidences']
        text_lengths = self.stats['text_lengths']
        
        # 确保数据长度一致
        min_len = min(len(confidences), len(text_lengths))
        confidences = confidences[:min_len]
        text_lengths = text_lengths[:min_len]
        
        return {
            'title': '置信度与文本长度关系',
            'type': 'scatter',
            'confidences': confidences,
            'text_lengths': text_lengths,
            'correlation': float(np.corrcoef(confidences, text_lengths)[0, 1]) if min_len > 1 else 0
        }
    
    def get_comprehensive_dashboard_data(self) -> Dict:
        """
        获取综合仪表板数据
        
        Returns:
            Dict: 完整的仪表板数据
        """
        return {
            'summary_stats': {
                'total_images': self.stats.get('total_images', 0),
                'total_detections': self.stats.get('total_detections', 0),
                'avg_confidence': round(self.stats.get('avg_confidence', 0), 4),
                'avg_detection_per_image': round(self.stats.get('avg_detection_per_image', 0), 2),
                'success_rate': 1.0  # 因为我们只加载成功的结果
            },
            'charts': {
                'confidence_distribution': self.get_confidence_distribution(),
                'text_length_distribution': self.get_text_length_distribution(),
                'detection_count_analysis': self.get_detection_count_analysis(),
                'text_frequency_analysis': self.get_text_frequency_analysis(),
                'confidence_vs_length_scatter': self.get_confidence_vs_length_scatter()
            },
            'metadata': {
                'generated_at': self._get_timestamp(),
                'data_source': 'OCR Batch Recognition Results',
                'processor_version': '1.0'
            }
        }
    
    def save_dashboard_data(self, output_file: str):
        """
        保存仪表板数据到JSON文件
        
        Args:
            output_file: 输出文件路径
        """
        dashboard_data = self.get_comprehensive_dashboard_data()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dashboard_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
        print(f"仪表板数据已保存到: {output_file}")
    
    def generate_simple_plots(self, output_dir: str = "./plots"):
        """
        生成简单的可视化图表
        
        Args:
            output_dir: 图表输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        
        # 1. 置信度分布直方图
        if self.stats.get('confidences'):
            plt.figure(figsize=(10, 6))
            plt.hist(self.stats['confidences'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('置信度分布', fontsize=14, fontweight='bold')
            plt.xlabel('置信度')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/confidence_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. 文本长度分布
        if self.stats.get('text_lengths'):
            plt.figure(figsize=(12, 6))
            plt.hist(self.stats['text_lengths'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
            plt.title('文本长度分布', fontsize=14, fontweight='bold')
            plt.xlabel('文本长度（字符数）')
            plt.ylabel('频次')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/text_length_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. 每张图片检测数量
        if self.stats.get('detection_counts'):
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, len(self.stats['detection_counts']) + 1), 
                    self.stats['detection_counts'], 
                    marker='o', linewidth=2, markersize=4)
            plt.title('每张图片检测文本数量', fontsize=14, fontweight='bold')
            plt.xlabel('图片序号')
            plt.ylabel('检测数量')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/detection_count_per_image.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"图表已保存到目录: {output_dir}")
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class NumpyEncoder(json.JSONEncoder):
    """处理numpy数据类型的JSON编码器"""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


# 便捷函数
def create_visualization_data(results: List[Dict], output_file: str = None) -> Dict:
    """
    便捷函数：从识别结果创建可视化数据
    
    Args:
        results: 识别结果列表
        output_file: 可选的输出文件路径
        
    Returns:
        Dict: 可视化数据
    """
    processor = OCRVisualizationProcessor()
    processor.load_results(results)
    
    dashboard_data = processor.get_comprehensive_dashboard_data()
    
    if output_file:
        processor.save_dashboard_data(output_file)
    
    return dashboard_data


def process_batch_json(json_file: str, output_dir: str = "./visualization_output") -> str:
    """
    便捷函数：处理批量识别JSON文件并生成可视化数据
    
    Args:
        json_file: 批量识别结果JSON文件
        output_dir: 输出目录
        
    Returns:
        str: 生成的仪表板数据文件路径
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    processor = OCRVisualizationProcessor()
    processor.load_from_json(json_file)
    
    # 保存仪表板数据
    dashboard_file = os.path.join(output_dir, "dashboard_data.json")
    processor.save_dashboard_data(dashboard_file)
    
    # 生成图表
    plots_dir = os.path.join(output_dir, "plots")
    processor.generate_simple_plots(plots_dir)
    
    return dashboard_file 