# asr_service.py

from fastapi import APIRouter
from pydantic import BaseModel
from funasr import AutoModel

router = APIRouter()

# ASR 模型加载
asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4")

class ASRRequest(BaseModel):
    audio_file: str

@router.post("/asr/")
async def auto_speech_recognition(audio_request: ASRRequest):
    """
    Performs automatic speech recognition (ASR) on the provided audio file.

    Args:
        audio_request (ASRRequest): A request object containing the following parameter:
            - audio_file (str): Path to the audio file that contains the speech to be recognized.

    Returns:
        dict: A dictionary containing the recognized text from the audio file.
            - text (str): The text recognized from the provided audio.
    """
    audio_input = audio_request.audio_file
    res = asr_model.generate(input=audio_input, batch_size_s=300)
    return {"text": res[0]['text']}