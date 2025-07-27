import os 
from funasr import AutoModel
from vllm import AsyncEngineArgs,AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig,snapshot_download
from fastapi import FastAPI, Request
from fastapi import UploadFile, File
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from prompt_utils import _build_prompt,remove_stop_words
import uuid
import json 
from pydantic import BaseModel
import asyncio
import edge_tts


# import os
# import random
# from pathlib import Path

# import cv2
# import numpy as np
# import torch
# from diffusers import AutoencoderKL, DDIMScheduler
# from omegaconf import OmegaConf
# from PIL import Image
# from echomimic.src.models.unet_2d_condition import UNet2DConditionModel
# from echomimic.src.models.unet_3d_echo import EchoUNet3DConditionModel
# from echomimic.src.models.whisper.audio2feature import load_audio_model
# from echomimic.src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
# from echomimic.src.utils.util import save_videos_grid, crop_and_pad
# from echomimic.src.models.face_locator import FaceLocator
# from moviepy.editor import VideoFileClip, AudioFileClip
# from facenet_pytorch import MTCNN
from echomimic import generate_video

# http接口服务
app=FastAPI()

class ASRRequest(BaseModel):
    audio_file: str  # 假设预期的字段是字符串

class TTSRequest(BaseModel):
    text: str
    output_file: str

class A2VRequest(BaseModel):
    uploaded_img: str
    uploaded_audio: str 
    width: int
    height: int 
    length: int
    seed: int
    facemask_dilation_ratio: float
    facecrop_dilation_ratio: float
    context_frames: int
    context_overlap: int
    cfg: int
    steps: int
    sample_rate: int 
    fps: int
    device: str


# -------------- ASR加载模型（可以在启动时加载以节省时间）-------------------
asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                      vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                      punc_model="ct-punc-c", punc_model_revision="v2.0.4")

# vLLM参数
# model_dir="qwen/Qwen-14B-Chat-Int4"
model_dir = "/data0/wujunfei/transformers/Qwen-7B-Chat-Int4"
tensor_parallel_size=1
gpu_memory_utilization=0.6
quantization='gptq'
dtype='float16'


# vLLM模型加载
def load_vllm():
    global generation_config,tokenizer,stop_words_ids,engine    
    # 模型下载
    # snapshot_download(model_dir)
    # 模型基础配置
    generation_config=GenerationConfig.from_pretrained(model_dir,trust_remote_code=True)
    # 加载分词器
    tokenizer=AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
    tokenizer.eos_token_id=generation_config.eos_token_id
    # 推理终止词
    stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
    # vLLM基础配置
    args=AsyncEngineArgs(model_dir)
    args.worker_use_ray=False
    args.engine_use_ray=False
    args.tokenizer=model_dir
    args.tensor_parallel_size=tensor_parallel_size
    args.trust_remote_code=True
    args.quantization=quantization
    args.gpu_memory_utilization=gpu_memory_utilization
    args.dtype=dtype
    args.max_num_seqs=20    # batch最大20条样本
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE']='True'
    engine=AsyncLLMEngine.from_engine_args(args)
    return generation_config,tokenizer,stop_words_ids,engine

generation_config,tokenizer,stop_words_ids,engine=load_vllm()

# 用户停止句匹配
def match_user_stop_words(response_token_ids,user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids)<len(stop_tokens):
            continue 
        if response_token_ids[-len(stop_tokens):]==stop_tokens:
            return True  # 命中停止句, 返回True
    return False




# ASR接口
@app.post("/asr/")
async def auto_speech_recognition(audio_request: ASRRequest):
    # 读取音频文件并传递给ASR模型
    audio_input = audio_request.audio_file
    print("Audio Input: ", audio_input)
    res = asr_model.generate(
        input=audio_input,
        batch_size_s=300
    )
    return {"text": res[0]['text']}


# chat对话接口
@app.post("/chat")
async def chat(request: Request):
    request=await request.json()
    
    query=request.get('query',None)
    history=request.get('history',[])
    system=request.get('system','You are a helpful assistant.')
    stream=request.get("stream",False)
    user_stop_words=request.get("user_stop_words",[])    # list[str]，用户自定义停止句，例如：['Observation: ', 'Action: ']定义了2个停止句，遇到任何一个都会停止
    print("query: ", query)

    if query is None:
        return Response(status_code=502,content='query is empty')

    # 用户停止词
    user_stop_tokens=[]
    for words in user_stop_words:
        user_stop_tokens.append(tokenizer.encode(words))
    
    # 构造prompt
    prompt_text,prompt_tokens=_build_prompt(generation_config,tokenizer,query,history=history,system=system)
        
    # vLLM请求配置
    sampling_params=SamplingParams(stop_token_ids=stop_words_ids, 
                                    early_stopping=False,
                                    top_p=generation_config.top_p,
                                    top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                    temperature=generation_config.temperature,
                                    repetition_penalty=generation_config.repetition_penalty,
                                    max_tokens=generation_config.max_new_tokens)
    # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
    request_id=str(uuid.uuid4().hex)
    # results_iter=engine.generate(prompt=None,sampling_params=sampling_params,prompt_token_ids=prompt_tokens,request_id=request_id)
    results_iter=engine.generate(inputs=prompt_text, sampling_params=sampling_params, request_id=request_id)

    # 流式返回，即迭代transformer的每一步推理结果并反复返回
    if stream:
        async def streaming_resp():
            async for result in results_iter:
                # 移除im_end,eos等系统停止词
                token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
                # 返回截止目前的tokens输出                
                text=tokenizer.decode(token_ids)
                yield (json.dumps({'text':text})+'\0').encode('utf-8')
                # 匹配用户停止词,终止推理
                if match_user_stop_words(token_ids,user_stop_tokens):
                    await engine.abort(request_id)   # 终止vllm后续推理
                    break
        return StreamingResponse(streaming_resp())

    # 整体一次性返回模式
    async for result in results_iter:
        # 移除im_end,eos等系统停止词
        token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
        # 返回截止目前的tokens输出                
        text=tokenizer.decode(token_ids)
        # 匹配用户停止词,终止推理
        if match_user_stop_words(token_ids,user_stop_tokens):
            await engine.abort(request_id)   # 终止vllm后续推理
            break
    
    return JSONResponse(ret)


@app.post("/a2v/")
async def Audio2Video(request: A2VRequest):
    """
    口唇驱动
    Args:
        request: {uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio, facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device}
    Return: 口唇驱动视频是否合成成功
    """
    final_output_path = process_video(
        request.uploaded_img, request.uploaded_audio, 
        request.width, request.height, request.length, request.seed, request.facemask_dilation_ratio, 
        request.facecrop_dilation_ratio, request.context_frames, request.context_overlap, request.cfg, request.steps, request.sample_rate, request.fps, request.device
    )
    return final_output_path


@app.post("/tts/")
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
    # asyncio.run(amain())
    await amain()
    return {"status": "success"}


if __name__=='__main__':
    uvicorn.run(app,
                host=None,
                port=24923,
                log_level="debug")