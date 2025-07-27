# chat_service.py

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from modelscope import AutoTokenizer, GenerationConfig
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from prompt_utils import _build_prompt, remove_stop_words
import uuid
import json
import os

router = APIRouter()


# Chat 模型加载
def load_vllm():
    global generation_config, tokenizer, stop_words_ids, engine
    model_dir = "Qwen/Qwen-7B-Chat-Int4"
    tensor_parallel_size = 1
    gpu_memory_utilization = 0.6
    quantization = 'gptq'
    dtype = 'float16'

    generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.eos_token_id = generation_config.eos_token_id
    stop_words_ids = [tokenizer.im_start_id, tokenizer.im_end_id, tokenizer.eos_token_id]
    args = AsyncEngineArgs(model_dir)
    args.worker_use_ray = False
    args.engine_use_ray = False
    args.tokenizer = model_dir
    args.tensor_parallel_size = tensor_parallel_size
    args.trust_remote_code = True
    args.quantization = quantization
    args.gpu_memory_utilization = gpu_memory_utilization
    args.dtype = dtype
    args.max_num_seqs = 20
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    engine = AsyncLLMEngine.from_engine_args(args)
    return generation_config, tokenizer, stop_words_ids, engine

generation_config, tokenizer, stop_words_ids, engine = load_vllm()


# 用户停止句匹配
def match_user_stop_words(response_token_ids,user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids)<len(stop_tokens):
            continue 
        if response_token_ids[-len(stop_tokens):]==stop_tokens:
            return True  # 命中停止句, 返回True
    return False


@router.post("/chat")
async def chat(request: Request):
    """
    Performs LLM interaction.

    Args:
        request (Request): A request object containing the following parameter:
            - query (str): The user query.
            - history (list): The history interaction.
            - system (str): System prompt.
            - stream (Boolean): Whether to output responses in stream.
            - user_stop_words (list)

    Returns:
       response: The streaming response or Json response.
    """
    request = await request.json()
    query = request.get('query', None)
    history = request.get('history', [])
    system = request.get('system', 'You are a helpful assistant.')
    stream = request.get("stream", False)
    user_stop_words = request.get("user_stop_words", [])


    if query is None:
        return Response(status_code=502, content='query is empty')

    user_stop_tokens = [tokenizer.encode(words) for words in user_stop_words]
    prompt_text, prompt_tokens = _build_prompt(generation_config, tokenizer, query, history=history, system=system)
    sampling_params = SamplingParams(stop_token_ids=stop_words_ids, 
                                     early_stopping=False,
                                     top_p=generation_config.top_p,
                                     top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                     temperature=generation_config.temperature,
                                     repetition_penalty=generation_config.repetition_penalty,
                                     max_tokens=generation_config.max_new_tokens)
    request_id = str(uuid.uuid4().hex)
    results_iter = engine.generate(inputs=prompt_text, sampling_params=sampling_params, request_id=request_id)

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
    
    return JSONResponse(text)