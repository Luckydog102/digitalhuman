# tts_service.py

from fastapi import APIRouter
from pydantic import BaseModel
import edge_tts
import asyncio

router = APIRouter()

class TTSRequest(BaseModel):
    text: str
    output_file: str

@router.post("/tts/")
async def text_to_speech(request: TTSRequest):
    """
    语音合成, 将语音合成并存储到指定路径
    Args:
        request: {"text": text, "output_file": output_file}
    Return: 语音合成是否合成成功
    """
    async def amain() -> None:
        communicate = edge_tts.Communicate(request.text, VOICE)
        await communicate.save(request.output_file)

    VOICE = 'zh-TW-HsiaoYuNeural'
    await amain()
    return {"status": "success"}