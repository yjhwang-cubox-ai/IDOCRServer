from dataclasses import dataclass
import logging
from datetime import datetime
import os
from typing import List, Dict

@dataclass
class ProcessingTime:
    image_name: str
    processing_time: float

class ProcessingTimeLogger:
    def __init__(self, log_dir: str, timestamp: str):
        self.log_dir = log_dir
        self.timestamp = timestamp
        self.processing_times: Dict[str, Dict] = {
            'idcard_detection': [],
            'text_detection': [],
            'text_recognition': [],
            'total_processing': []
        }
        self.text_box_counts: Dict[str, int] = []
        self._setup_logger()        
    
    def _setup_logger(self) -> None:
        self.loggers = {}
        for process_name in self.processing_times.keys():
            logger = logging.getLogger(process_name)
            logger.setLevel(logging.INFO)

            #파일 핸들러 설정
            log_file = os.path.join(self.log_dir, f'{process_name}_{self.timestamp}.txt')
            file_handler = logging.FileHandler(log_file, mode='w')
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            self.loggers[process_name] = logger
            
            #로그 파일 헤더 작성
            logger.info(f"{process_name.replace('_', ' ').title()} Times")
            logger.info("="*50 + "\n")
            
        # 텍스트 박스 카운트 로거 설정
        self.box_logger = logging.getLogger('text_box_count')
        self.box_logger.setLevel(logging.INFO)
        box_log_file = os.path.join(self.log_dir, f'text_box_counts_{self.timestamp}.txt')
        box_handler = logging.FileHandler(box_log_file, mode='w')
        box_handler.setFormatter(formatter)
        self.box_logger.addHandler(box_handler)
        self.box_logger.info("Text Box Count Per Image")
        self.box_logger.info("="*50 + "\n")
    
    def log_processing_time(self, process_name: str, image_name: str, processing_time: float):
        """처리 시간 로깅"""
        self.processing_times[process_name].append(ProcessingTime(image_name, processing_time))
        self.loggers[process_name].info(f"Image: {image_name} - {processing_time:.3f} seconds")
    
    def log_text_box_count(self, image_name: str, count: int):
        """텍스트 박스 개수 로깅"""
        self.text_box_counts.append(count)
        self.box_logger.info(f"Image: {image_name} - {count} text boxes detected")
    
    def summarize_results(self):
        """결과 요약 및 출력"""
        print("\nProcessing Results:")
        print("="*50)
        
        # 처리 시간 평균 계산 및 출력
        for process_name, times in self.processing_times.items():
            if not times:
                continue
                
            avg_time = sum(t.processing_time for t in times) / len(times)
            max_time = max(t.processing_time for t in times)
            min_time = min(t.processing_time for t in times)
            
            # 로그 파일에 통계 기록
            self.loggers[process_name].info("\nProcessing Time Statistics")
            self.loggers[process_name].info("="*50)
            self.loggers[process_name].info(f"Average time: {avg_time:.3f} seconds")
            self.loggers[process_name].info(f"Maximum time: {max_time:.3f} seconds")
            self.loggers[process_name].info(f"Minimum time: {min_time:.3f} seconds")
            
            # 콘솔에 출력
            print(f"\n{process_name.replace('_', ' ').title()}:")
            print(f"  Average: {avg_time:.3f} seconds")
            print(f"  Maximum: {max_time:.3f} seconds")
            print(f"  Minimum: {min_time:.3f} seconds")
        
        # 텍스트 박스 통계 계산 및 출력
        if self.text_box_counts:
            avg_boxes = sum(self.text_box_counts) / len(self.text_box_counts)
            max_boxes = max(self.text_box_counts)
            min_boxes = min(self.text_box_counts)
            
            print("\nText Box Statistics:")
            print(f"  Average boxes per image: {avg_boxes:.2f}")
            print(f"  Maximum boxes in an image: {max_boxes}")
            print(f"  Minimum boxes in an image: {min_boxes}")
            print(f"  Total images processed: {len(self.text_box_counts)}")

class LayoutTimeLogger:
    def __init__(self, log_dir: str, timestamp: str):
        self.log_dir = log_dir
        self.timestamp = timestamp
        self.processing_times: Dict[str, Dict] = {
            'layout_analysis': []
        }
        self._setup_logger()        
    
    def _setup_logger(self) -> None:
        self.loggers = {}
        for process_name in self.processing_times.keys():
            logger = logging.getLogger(process_name)
            logger.setLevel(logging.INFO)

            #파일 핸들러 설정
            log_file = os.path.join(self.log_dir, f'{process_name}_{self.timestamp}.txt')
            file_handler = logging.FileHandler(log_file, mode='w')
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            self.loggers[process_name] = logger
            
            #로그 파일 헤더 작성
            logger.info(f"{process_name.replace('_', ' ').title()} Times")
            logger.info("="*50 + "\n")
    
    def log_processing_time(self, process_name: str, image_name: str, processing_time: float):
        """처리 시간 로깅"""
        self.processing_times[process_name].append(ProcessingTime(image_name, processing_time))
        self.loggers[process_name].info(f"Image: {image_name} - {processing_time:.3f} seconds")
    
    def summarize_results(self):
        """결과 요약 및 출력"""
        print("\nProcessing Results:")
        print("="*50)
        
        # 처리 시간 평균 계산 및 출력
        for process_name, times in self.processing_times.items():
            if not times:
                continue
                
            avg_time = sum(t.processing_time for t in times) / len(times)
            max_time = max(t.processing_time for t in times)
            min_time = min(t.processing_time for t in times)
            
            # 로그 파일에 통계 기록
            self.loggers[process_name].info("\nProcessing Time Statistics")
            self.loggers[process_name].info("="*50)
            self.loggers[process_name].info(f"Average time: {avg_time:.3f} seconds")
            self.loggers[process_name].info(f"Maximum time: {max_time:.3f} seconds")
            self.loggers[process_name].info(f"Minimum time: {min_time:.3f} seconds")
            
            # 콘솔에 출력
            print(f"\n{process_name.replace('_', ' ').title()}:")
            print(f"  Average: {avg_time:.3f} seconds")
            print(f"  Maximum: {max_time:.3f} seconds")
            print(f"  Minimum: {min_time:.3f} seconds")