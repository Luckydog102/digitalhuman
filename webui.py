import gradio as gr
import requests
import json
import soundfile as sf

MAX_HISTORY_LEN=50
ASR_API_URL = 'http://localhost:8000/asr/asr/'
QA_API_URL = 'http://localhost:8000/chat/chat/'
TTS_API_URL = 'http://localhost:8000/tts/tts/'
A2V_API_URL = 'http://localhost:8000/a2v/a2v/'
VOICE_CLONE_API_URL = 'http://localhost:8000/voice_clone/voice_clone/'


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


# 处理音频到文本的请求
def audio_to_text(audio):
    sample_rate, audio_data = audio
    # print("Type: ", type(audio_data))
    # print("Shape: ", audio_data.shape)
    # 将 NumPy 数组保存为 WAV 文件
    file_path = "temp/temp_audio.wav"
    sf.write(file_path, audio_data, sample_rate)

    response = requests.post(ASR_API_URL, json={"audio_file": file_path})
    response_data = response.json()
    print("ASR Response Data: ", response_data)
    return response_data.get("text", "")


def chat_streaming(query,history, stream = False):
    # 调用api_server
    response=requests.post(QA_API_URL,json={
        'query': query,
        'stream': stream,
        'history':history
    }, stream = stream)
    
    # 流式读取http response body, 按\0分割
    for chunk in response.iter_lines(chunk_size=8192,decode_unicode=False,delimiter=b"\0"):
        if chunk:
            data=json.loads(chunk.decode('utf-8'))
            text=data["text"].rstrip('\r\n') # 确保末尾无换行
            yield text


# 处理文本到语音的请求
def text_to_speech(text):
    output_file = "temp/output.wav"
    requests.post(TTS_API_URL, json={"text": text, "output_file": output_file})
    return output_file


def voice_clone(text, reference_speaker):
    output_file = "temp/output.wav"
    requests.post(VOICE_CLONE_API_URL, json={"text": text, "output_file": output_file, "reference_speaker": reference_speaker})
    return output_file


# 调用请求函数
def chat_audio_input(audio, history):
    # asr
    query = audio_to_text(audio)

    # qa
    for response in chat_streaming(query,history, stream=True):
        final_response = response
        yield '', history + [(query, response)], None
        
    # TTS: 将生成的文本转换为语音
    tts_audio = text_to_speech(response)
    yield '', history + [(query, response)], tts_audio

    history.append((query,response))
    while len(history) > MAX_HISTORY_LEN:
        history.pop(0)
    

def chat_text_input(query, history):
    # for response in chat_streaming(query,history):
    #     yield '', history+[(query,response)]

    for response in chat_streaming(query,history, stream=True):
        final_response = response
        yield '', history + [(query, response)], None

    # TTS: 将生成的文本转换为语音
    tts_audio = text_to_speech(response)
    yield '', history + [(query, response)], tts_audio

    history.append((query,response))
    while len(history) > MAX_HISTORY_LEN:
        history.pop(0)


def chat_audio_input_clone(audio, history, reference_speaker):
    # asr
    query = audio_to_text(audio)

    # qa
    for response in chat_streaming(query,history, stream=True):
        final_response = response
        yield '', history + [(query, response)], None
        
    #  Voice Clone: 将生成的文本转换为语音，并根据参考语音进行复刻
    tts_audio = voice_clone(response, reference_speaker)
    yield '', history + [(query, response)], tts_audio

    history.append((query,response))
    while len(history) > MAX_HISTORY_LEN:
        history.pop(0)


def chat_text_input_clone(query, history, reference_speaker):
    for response in chat_streaming(query,history, stream=True):
        final_response = response
        yield '', history + [(query, response)], None

    #  Voice Clone: 将生成的文本转换为语音，并根据参考语音进行复刻
    tts_audio = voice_clone(response, reference_speaker)
    yield '', history + [(query, response)], tts_audio 

    history.append((query,response))
    while len(history) > MAX_HISTORY_LEN:
        history.pop(0)


def generate_video(uploaded_img, uploaded_audio, width, height, length, seed, facemask_dilation_ratio, facecrop_dilation_ratio, context_frames, context_overlap, cfg, steps, sample_rate, fps, device):
    response = requests.post(A2V_API_URL, 
                            json={
                                "uploaded_img": uploaded_img,
                                "uploaded_audio": uploaded_audio,
                                "width": width,
                                "height": height, 
                                "length": length,
                                "seed": seed,
                                "facemask_dilation_ratio": facemask_dilation_ratio,
                                "facecrop_dilation_ratio": facecrop_dilation_ratio,
                                "context_frames": context_frames,
                                "context_overlap": context_overlap,
                                "cfg": cfg,
                                "steps": steps,
                                "sample_rate": sample_rate, 
                                "fps": fps,
                                "device": device })
    response_data = response.json()
    print("Audio2Video Response Data: ", response_data)
    return response_data.get("output_path", None)



with gr.Blocks(css='.qwen-logo img {height:100px; width:300px; margin:0 auto;}') as app:
    with gr.Column():
        logo_img = gr.Image('assets/yidong.webp', elem_classes='qwen-logo')
    
    with gr.Row():
        with gr.Column():
            uploaded_img = gr.Image(type="filepath", label="虚拟数字人形象", sources=["upload"])
        with gr.Column():
            audio_reference = gr.Audio(type='filepath', sources=["upload"], label="参考音频")  
            

    with gr.Column():
        chatbot = gr.Chatbot(label='通义千问7B-Chat-Int4')


    with gr.Row():
        with gr.Column(scale=2):
            query_box = gr.Textbox(label='提问', autofocus=True, lines=5)
        with gr.Column(scale=1, min_width=200):
            submit_text_btn = gr.Button(value='文本问题提交')
    
    with gr.Row():
        with gr.Column(scale=2):
            audio_input = gr.Audio(sources=["microphone"], label="语音输入")  # 仅用microphone输入，不包含上传文件
        with gr.Column(scale=1, min_width=200):
            submit_audio_btn = gr.Button(value='音频问题提交')

    
    with gr.Row():
        with gr.Column():     
            audio_output = gr.Audio(type="filepath", label="语音回答", sources=[])
        with gr.Column():
            output_video = gr.Video() 

    with gr.Accordion("Configuration合成参数", open=False):
        width = gr.Slider(label="Width(建议512)", minimum=128, maximum=1024, value=default_values['width'])
        height = gr.Slider(label='Height(建议512)', minimum=128, maximum=1024, value=default_values['height'])
        length = gr.Slider(label='Length', minimum=100, maximum=5000, value=default_values['length'])
        seed = gr.Slider(label='Seed', minimum=0, maximum=10000, value=default_values['seed'])
        facemask_dilation_ratio = gr.Slider(label='Facemask Dilation Ratio', minimum=0.0, maximum=1.0, step=0.01, value=default_values['facemask_dilation_ratio'])
        facecrop_dilation_ratio = gr.Slider(label='Facecrop Dilation Ratio(人脸裁剪区域大小)', minimum=0.0, maximum=1.0, step=0.01, value=default_values['facecrop_dilation_ratio'])
        context_frames = gr.Slider(label='Context Frames', minimum=0, maximum=50, step=1, value=default_values['context_frames'])
        context_overlap = gr.Slider(label='Context Overlap', minimum=0, maximum=10, step=1, value=default_values['context_overlap'])
        cfg = gr.Slider(label='CFG', minimum=0.0, maximum=10.0, step=0.1, value=default_values['cfg'])
        steps = gr.Slider(label='Steps(迭代步数，越大越慢，但画面更稳定)', minimum=1, maximum=100, step=1, value=default_values['steps'])
        sample_rate = gr.Slider(label='Sample Rate', minimum=8000, maximum=48000, step=1000, value=default_values['sample_rate'])
        fps = gr.Slider(label='FPS', minimum=1, maximum=60, step=1, value=default_values['fps'])
        device = gr.Radio(label='Device', choices=['cuda', 'cpu'], value=default_values['device'])

    with gr.Row():
        generate_button = gr.Button("Generate Video开始合成视频")
        clear_btn = gr.ClearButton([query_box, chatbot, audio_input, audio_output, output_video], value='清空对话历史')

    
    # Edge TTS
    # submit_audio_btn.click(chat_audio_input, [audio_input, chatbot], [query_box, chatbot, audio_output])
    # submit_text_btn.click(chat_text_input, [query_box,chatbot], [query_box, chatbot, audio_output])

    # OpenVoice

    submit_audio_btn.click(chat_audio_input_clone, [audio_input, chatbot, audio_reference], [query_box, chatbot, audio_output])
    submit_text_btn.click(chat_text_input_clone, [query_box,chatbot, audio_reference], [query_box, chatbot, audio_output])
    print("Image Type: ", type(uploaded_img))
    print("Audio Type: ", type(audio_input))
    
    generate_button.click(
        generate_video,
        inputs=[
            uploaded_img,
            audio_output,
            width,
            height,
            length,
            seed,
            facemask_dilation_ratio,
            facecrop_dilation_ratio,
            context_frames,
            context_overlap,
            cfg,
            steps,
            sample_rate,
            fps,
            device
        ],
        outputs=output_video
    )


if __name__ == "__main__":
    app.queue(200)  # 请求队列
    app.launch(server_name='127.0.0.1', server_port=2024, max_threads=500) # 线程池