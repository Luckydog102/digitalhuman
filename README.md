# AIGC相关技术研究之数字人技术


## 实现方案
1. 数字人形象与背景生成（任务一）：
    - Stable diffusion
    - 文生图
    - 图生图
2. 数字人语音复刻（任务二）：GPT-SoVITS，OpenVoice
3. 数字人口唇驱动（任务四）：
    - SadTalker口唇匹配
    - EchoMimic口唇及头部运动
    - Audio2Face实现驱动UE5数字人角色动画
4. 数字人交互问答（任务五）：
    - 语音识别: Paraformer
    - 大模型：Qwen-7B-Int4
    - 推理加速：vLLM


## Python环境配置
- Tested System Environment: Centos 7.2/Ubuntu 22.04, Cuda >= 11.7
- Tested GPUs: A100(80G) / RTX4090D (24G) / V100(16G)


创建Conda虚拟环境（推荐）:
```bash
  conda create -n digital_human
  conda activate digital_human
```

使用 `pip` 命令安装依赖包：
```bash
  pip install -r requirements.txt
```

## 运行

```bash
cd project
uvicorn main:app --reload               # 启动服务端
python webui.py                         # 启动前端
```

## 成果展示

1. `project/task_1/`路径包含生成的虚拟数字人形象与背景。
2. `交付物/` 路径包含任务1，2，4，5的交付物内容。

整体项目展示如下：

1. （任务一）数字人形象生成后，上传图片
2. （任务五）输入文本问题与语音问题，大模型生成文本回答并流式显示，生成语音回答
3. （任务二）根据合成语音与参考语音，输出语音复刻后的语音回答
4. （任务四）基于数字人形象、回答语音，生成数字人讲话视频
