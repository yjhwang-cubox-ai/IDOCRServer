from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import time
from typing import List
import uvicorn
from processors.ocr_processor import OCRProcessor
from processors.layout_processor import LayoutProcessor
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 한 번만 인스턴스 생성
    app.state.ocr_processor = OCRProcessor()
    app.state.layout_processor = LayoutProcessor()
    yield
    # 서버 종료 시 필요한 정리 작업이 있다면 여기서 수행
    # 예: app.state.ocr_processor.cleanup() 등

app = FastAPI(title="OCR API Server", lifespan=lifespan)

# CORS 미들웨어 설정 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # 실제 Svelte 앱이 실행되는 포트로 수정
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OCRData(BaseModel):
    head: str
    name: str
    idnumber: str
    address: str
    Issued_date: str
    Issuer: str

class APIResponse(BaseModel):
    code: int
    message: str
    data: OCRData
    processing_time: float

def run_inference(image_array: np.ndarray, ocr_processor: OCRProcessor, layout_processor: LayoutProcessor) -> List[str]:
    ocr_output = ocr_processor.process_single_image(image_array)
    layout_output = layout_processor.predict(
        images=ocr_output['aligned_img'],
        texts=ocr_output['texts'],
        bboxes=ocr_output['text_boxes']
    )
    
    return layout_output

@app.post("/ocr/")
async def perform_ocr(request:Request, file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 저장된 인스턴스를 가져오기
        ocr_processor = request.app.state.ocr_processor
        layout_processor = request.app.state.layout_processor

        # 추론 실행
        start_time = time.time()
        output = run_inference(image, ocr_processor, layout_processor)
        infer_time = time.time() - start_time
        
        data = OCRData(
            head=output['head'],
            name=output['name'],
            idnumber=output['idnumber'],
            address=output['address'],
            Issued_date=output['Issued_date'],
            Issuer=output['Issuer'],
        )
        
        # 결과 반환
        return APIResponse(code=200, message="success", data=data, processing_time=infer_time)
    
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