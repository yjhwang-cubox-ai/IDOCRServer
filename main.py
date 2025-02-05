from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from typing import List
import uvicorn
from processors.ocr_processor import OCRProcessor
from processors.layout_processor import LayoutProcessor

app = FastAPI(title="OCR API Server")

# CORS 미들웨어 설정 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # 실제 Svelte 앱이 실행되는 포트로 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OCRResponse(BaseModel):
    status: str
    text: List[str]  # OCR 텍스트 결과
    processing_time: float  # 처리 시간

def run_inference(image_array: np.ndarray) -> List[str]:    
    ocr_processor = OCRProcessor()
    layout_processor = LayoutProcessor()

    ocr_output = ocr_processor.process_single_image(image_array)
    layout_output = layout_processor.predict(
        images=ocr_output['aligned_img'],
        texts=ocr_output['texts'],
        bboxes=ocr_output['text_boxes']
    )
    
    return layout_output

@app.post("/ocr/")
async def perform_ocr(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 추론 실행
        results = run_inference(image)
        
        # 결과 반환
        return OCRResponse(
            status="success",
            text=results['texts'],
            processing_time=results['processing_time']
        )
    
    except Exception as e:
        return JSONResponse(
            content={
                "status": "error",
                "message": str(e),
                "detail": {
                    "error_type": type(e).__name__,
                    "error_location": "perform_ocr"
                }
            },
            status_code=500
        )

# 서버 상태 확인용 엔드포인트
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":    
    uvicorn.run(app, host="0.0.0.0", port=8000)