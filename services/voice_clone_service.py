# voice_clone_service.py

from fastapi import APIRouter
from pydantic import BaseModel
import edge_tts
import asyncio
import torch
from openvoice import se_extractor
from openvoice.api import BaseSpeakerTTS, ToneColorConverter

router = APIRouter()


class TTSRequest(BaseModel):
    reference_speaker: str
    text: str
    output_file: str



@router.post("/voice_clone/")
async def clone_voice(request: TTSRequest):
    """
    Perform voice cloning based on the reference voice and input text.
    Args:
        request (TTSRequest): A request object containing the following parameter:
            - reference_speaker (str): Path to the audio file that contains the reference speaker voice.
            - text (str): The text that needed to be transformed into speech.
            - output_file (str): Path to the output audio file that.
       
    Returns: 
        dict: A dictionary containing the cloning status.
            - status (str): The cloning status.
    """
    
    ckpt_base = './services/OpenVoice/checkpoints/base_speakers/ZH'
    ckpt_converter = './services/OpenVoice/checkpoints/converter'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    output_dir = 'temp'

    # 加载色调转换器和基础说话人TTS模型
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    # 加载预训练的基础说话人特征
    source_se = torch.load(f'{ckpt_base}/zh_default_se.pth').to(device)

    # 提取目标说话人的特征向量
    target_se, _ = se_extractor.get_se(request.reference_speaker, tone_color_converter, target_dir='processed', vad=True)

    # 生成基础说话人语音
    src_path = f'{output_dir}/tmp.wav'
    base_speaker_tts.tts(request.text, src_path, speaker='default', language='Chinese', speed=1.0)


    # 语音转换为目标说话人的声音
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=request.output_file,
        message=encode_message
    )

    return {"status": "success"}

# ckpt_base = 'OpenVoice/checkpoints/base_speakers/ZH'
# ckpt_converter = 'OpenVoice/checkpoints/converter'
# device="cuda:0" if torch.cuda.is_available() else "cpu"
# output_dir = 'temp'

# tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
# tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
# base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

# source_se = torch.load(f'{ckpt_base}/zh_default_se.pth').to(device)
# save_path = f'{output_dir}/output_chinese.wav'

# reference_speaker = 'OpenVoice/resources/demo_speaker1.mp3' # This is the voice you want to clone
# target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed', vad=True)

# # Run the base speaker tts
# # text = "今天天气真好，我们一起出去吃饭吧。"
# text = "哈哈哈哈哈。请于2023年11月1日起登录福建“海纳百川”人才网引进生专区(http://fjhnbc.hxrc.com/yjs)注册报名，报名时间截止至11月21日18:00。"
# src_path = f'{output_dir}/tmp.wav'
# base_speaker_tts.tts(text, src_path, speaker='default', language='Chinese', speed=1.0)

# # Run the tone color converter
# encode_message = "@MyShell"
# tone_color_converter.convert(
#     audio_src_path=src_path, 
#     src_se=source_se, 
#     tgt_se=target_se, 
#     output_path=save_path,
#     message=encode_message)