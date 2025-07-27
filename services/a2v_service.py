#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import os
import sys
import random
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_echo import EchoUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echo_mimic_acc import Audio2VideoPipeline
from src.utils.util import save_videos_grid, crop_and_pad
from src.models.face_locator import FaceLocator
from moviepy.editor import VideoFileClip, AudioFileClip
from facenet_pytorch import MTCNN
import argparse
from fastapi import APIRouter, Request
from pydantic import BaseModel


router = APIRouter()

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
    

default_values = {
    "width": 512,
    "height": 512,
    "length": 1200,
    "seed": 420,
    "facemask_dilation_ratio": 0.1,
    "facecrop_dilation_ratio": 0.7,
    "context_frames": 12,
    "context_overlap": 3,
    "cfg": 1,
    "steps": 6,
    "sample_rate": 16000,
    "fps": 24,
    "device": "cuda"
}

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print(
        "please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.getenv('PATH'):
    print("add ffmpeg to path")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

config_path = "./services/echomimic/configs/prompts/animation_digital_human.yaml"
config = OmegaConf.load(config_path)
if config.weight_dtype == "fp16":
    weight_dtype = torch.float16
else:
    weight_dtype = torch.float32

device = "cuda"
if not torch.cuda.is_available():
    print("CUDA is not available")
    device = "cpu"

inference_config_path = config.inference_config
infer_config = OmegaConf.load(inference_config_path)

############# model_init started #############
## vae init
vae = AutoencoderKL.from_pretrained(config.pretrained_vae_path).to(device, dtype=weight_dtype)

## reference net init
reference_unet = UNet2DConditionModel.from_pretrained(
    config.pretrained_base_model_path,
    subfolder="unet",
).to(dtype=weight_dtype, device=device)
reference_unet.load_state_dict(torch.load(config.reference_unet_path, map_location="cpu"))

## denoising net init
if os.path.exists(config.motion_module_path):
    ### stage1 + stage2
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        config.motion_module_path,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)
else:
    ### only stage1
    denoising_unet = EchoUNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "cross_attention_dim": infer_config.unet_additional_kwargs.cross_attention_dim
        }
    ).to(dtype=weight_dtype, device=device)

print(config.denoising_unet_path)
denoising_unet.load_state_dict(torch.load(config.denoising_unet_path, map_location="cpu"), strict=False)

## face locator init
face_locator = FaceLocator(320, conditioning_channels=1, block_out_channels=(16, 32, 96, 256)).to(dtype=weight_dtype,
                                                                                                  device=device)
face_locator.load_state_dict(torch.load(config.face_locator_path))

## load audio processor params
audio_processor = load_audio_model(model_path=config.audio_model_path, device=device)

## load face detector params
face_detector = MTCNN(image_size=320, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709,
                      post_process=True, device=device)

############# model_init finished #############

sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
scheduler = DDIMScheduler(**sched_kwargs)

pipe = Audio2VideoPipeline(
    vae=vae,
    reference_unet=reference_unet,
    denoising_unet=denoising_unet,
    audio_guider=audio_processor,
    face_locator=face_locator,
    scheduler=scheduler,
).to(device, dtype=weight_dtype)


def select_face(det_bboxes, probs):
    ## max face from faces that the prob is above 0.8
    ## box: xyxy
    if det_bboxes is None or probs is None:
        return None
    filtered_bboxes = []
    for bbox_i in range(len(det_bboxes)):
        if probs[bbox_i] > 0.8:
            filtered_bboxes.append(det_bboxes[bbox_i])
    if len(filtered_bboxes) == 0:
        return None

    sorted_bboxes = sorted(filtered_bboxes, key=lambda x:(x[3]-x[1]) * (x[2] - x[0]), reverse=True)
    return sorted_bboxes[0]


def process_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio,
                  facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device):
    if seed is not None and seed > -1:
        generator = torch.manual_seed(seed)
    else:
        generator = torch.manual_seed(random.randint(100, 1000000))
    #### face musk prepare
    with open(uploaded_img, 'rb') as f:
        img_data = f.read()
    img_array = np.frombuffer(img_data, np.uint8)
    face_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # 检查图片是否成功读取
    if face_img is not None:
        print("Image read successfully")
    else:
        print("Image read Failed")
    # face_img = cv2.imread(uploaded_img) 原写法遇到中文会报错
    face_mask = np.zeros((face_img.shape[0], face_img.shape[1])).astype('uint8')
    det_bboxes, probs = face_detector.detect(face_img)
    select_bbox = select_face(det_bboxes, probs)
    if select_bbox is None:
        face_mask[:, :] = 255
    else:
        xyxy = select_bbox[:4]
        xyxy = np.round(xyxy).astype('int')
        rb, re, cb, ce = xyxy[1], xyxy[3], xyxy[0], xyxy[2]
        r_pad = int((re - rb) * facemask_dilation_ratio)
        c_pad = int((ce - cb) * facemask_dilation_ratio)
        face_mask[rb - r_pad: re + r_pad, cb - c_pad: ce + c_pad] = 255

        #### face crop
        r_pad_crop = int((re - rb) * facecrop_dilation_ratio)
        c_pad_crop = int((ce - cb) * facecrop_dilation_ratio)
        crop_rect = [max(0, cb - c_pad_crop), max(0, rb - r_pad_crop), min(ce + c_pad_crop, face_img.shape[1]),
                     min(re + r_pad_crop, face_img.shape[0])]
        face_img,_ = crop_and_pad(face_img, crop_rect)
        face_mask,_ = crop_and_pad(face_mask, crop_rect)
        dsize=(width, height)
        face_img = cv2.resize(face_img, dsize)
        face_mask = cv2.resize(face_mask, dsize)

    ref_image_pil = Image.fromarray(face_img[:, :, [2, 1, 0]])
    face_mask_tensor = torch.Tensor(face_mask).to(dtype=weight_dtype, device=device).unsqueeze(0).unsqueeze(
        0).unsqueeze(0) / 255.0

    video = pipe(
        ref_image_pil,
        uploaded_audio,
        face_mask_tensor,
        width,
        height,
        length,
        steps,
        cfg,
        generator=generator,
        audio_sample_rate=sample_rate,
        context_frames=context_frames,
        fps=fps,
        context_overlap=context_overlap
    ).videos

    save_dir = Path("temp/")
    save_dir.mkdir(exist_ok=True, parents=True)
    output_video_path = save_dir / "output_video.mp4"
    save_videos_grid(video, str(output_video_path), n_rows=1, fps=fps)

    video_clip = VideoFileClip(str(output_video_path))
    audio_clip = AudioFileClip(uploaded_audio)
    final_output_path = save_dir / "output_video_with_audio.mp4"
    video_clip = video_clip.set_audio(audio_clip)
    video_clip.write_videofile(str(final_output_path), codec="libx264", audio_codec="aac")

    return final_output_path


@router.post("/a2v/")
def Audio2Video(request: A2VRequest):
    """
    Converts an audio file into a video file using the provided image, audio and parameters.

    Args:
        request (A2VRequest): A request object containing the following parameters:
            - uploaded_img (str): Path to the uploaded image to be used in the video.
            - uploaded_audio (str): Path to the uploaded audio file to be synchronized with the image.
            - width (int): The width of the output video.
            - height (int): The height of the output video.
            - length (float): The desired length of the output video in seconds.
            - seed (int): Random seed for reproducibility.
            - facemask_dilation_ratio (float): Ratio to control the dilation of the facial mask during processing.
            - facecrop_dilation_ratio (float): Ratio to control the dilation of the face crop area.
            - context_frames (int): Number of context frames to be used in the video generation.
            - context_overlap (float): Overlap ratio between context frames.
            - cfg (float): Config scale parameter for controlling the image generation.
            - steps (int): Number of steps for the video generation process.
            - sample_rate (int): Audio sample rate to be used in the video.
            - fps (int): Frames per second for the output video.
            - device (str): The device (e.g., "cpu", "cuda") to be used for processing.

    Returns:
        dict: A dictionary containing the path to the generated video file.
            - output_path (str): The path to the final output video file.
    """
    final_output_path = process_video(
        request.uploaded_img, request.uploaded_audio, 
        request.width, request.height, request.length, request.seed, request.facemask_dilation_ratio, 
        request.facecrop_dilation_ratio, request.context_frames, request.context_overlap, request.cfg, request.steps, request.sample_rate, request.fps, 
        request.device
    )
    return {"output_path": final_output_path}

