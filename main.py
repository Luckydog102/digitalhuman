# main.py

import os
import sys

# 添加项目路径到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 添加自定义路径到系统路径中
custom_path_1 = "./services/echomimic"
custom_path_2 = "./services/OpenVoice"
if custom_path_1 not in sys.path:
    sys.path.append(custom_path_1)
if custom_path_2 not in sys.path:
    sys.path.append(custom_path_2)
# print("System Path: ", sys.path)

from fastapi import FastAPI
import uvicorn
from services.asr_service import router as asr_router
from services.chat_service import router as chat_router
from services.a2v_service import router as a2v_router
from services.voice_clone_service import router as voice_clone_router

app = FastAPI()

# 将各个服务的路由注册到主应用中
app.include_router(asr_router, prefix="/asr")
app.include_router(chat_router, prefix="/chat")
app.include_router(a2v_router, prefix="/a2v")
app.include_router(voice_clone_router, prefix="/voice_clone")


if __name__ == "__main__":
    uvicorn.run(app,
            host=None,
            port=24923,
            log_level="debug")


from services.tts_service import router as tts_router
app.include_router(tts_router, prefix="/tts")