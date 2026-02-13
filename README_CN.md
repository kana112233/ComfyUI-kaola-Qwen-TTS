# ComfyUI-Qwen3-TTS

这是一个为 ComfyUI 开发的 **Qwen3-TTS** 节点插件，支持阿里巴巴 Qwen3-TTS 系列模型的语音生成功能。

支持功能：
- **Custom Voice (自定义声音)**：使用预置的高质量音色。
- **Voice Design (声音设计)**：通过文字描述创造全新的声音。
- **Voice Clone (声音复刻)**：克隆参考音频的声音。

## 📦 安装 (Installation)

1.  进入 ComfyUI 的 `custom_nodes` 目录：
    ```bash
    cd ComfyUI/custom_nodes/
    ```
2.  克隆本项目：
    ```bash
    git clone https://github.com/Startgame/ComfyUI-kaola-Qwen-TTS.git
    ```
3.  安装依赖：
    ```bash
    cd ComfyUI-kaola-Qwen-TTS
    pip install -r requirements.txt
    ```

## 📥 模型下载 (Model Download)

节点支持自动下载模型（缓存到 HuggingFace 默认目录），但建议您手动下载以更好地管理。

请将模型放置在以下目录结构中：

```
ComfyUI/models/qwen3_tts/
```

如果目录不存在，请手动创建。

推荐的模型：
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (推荐用于 Custom Voice)
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (推荐用于 Voice Design)
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (推荐用于 Voice Clone)

## 🧩 节点说明 (Nodes)

### 1. Qwen3TTS Loader (模型加载器)
- **作用**：加载模型权重。
- **注意**：请务必选择与后续生成任务匹配的模型类型！
    - 想用 Custom Voice -> 加载 `CustomVoice` 模型。
    - 想用 Voice Design -> 加载 `VoiceDesign` 模型。
    - 想用 Voice Clone -> 加载 `Base` 模型。

### 2. Qwen3TTS Custom Voice (自定义声音)
- **参数**：
    - `text`: 要朗读的文本。
    - `speaker`: 选择预置说话人（如 Vivian, Ryan 等）。
    - `instruct`: (可选) 情感/语气描述，如 "Happy", "Sad"。
    - `language`: 输出语言（Auto 为自动识别）。

### 3. Qwen3TTS Voice Design (声音设计)
- **参数**：
    - `instruct`: **必须填写**。用自然语言描述你想要的声音，例如："A deep, husky male voice"（低沉沙哑的男声）。
    - `text`: 要朗读的文本。

### 4. Qwen3TTS Voice Clone (声音复刻)
- **参数**：
    - `ref_audio`: 参考音频。可以通过 ComfyUI 的音频输入连接，或者填写本地文件路径 `ref_audio_path`。
    - `ref_text`: **必须填写**参考音频对应的文本内容（逐字稿）。这能显著提升克隆相似度。
    - `text`: 要朗读的文本。
    - `x_vector_only`: 若勾选，则仅使用声纹特征（此时可以不填 `ref_text`），但效果通常不如包含文本的克隆模式好。

## ⚠️ 注意事项 (Notes)

1.  **显存要求**：1.7B 模型大约需要 4GB+ 显存。如果显存不足，尝试使用 0.6B 版本。
2.  **Flash Attention**：如果你使用 NVIDIA 显卡，建议安装 `flash-attn` 库以获得更快的推理速度。
3.  **模型匹配**：**切勿**将 `Base` 模型连接到 `CustomVoice` 节点，或者将 `CustomVoice` 模型连接到 `VoiceDesign` 节点，这会导致报错。请确保 Loader 和 Generator 类型一致。

## 示例 (Examples)
项目根目录下的 `examples/` 文件夹中包含以下工作流：

1.  `examples/qwen3_tts_example_workflow.json`：包含基础功能的示例。
2.  `examples/qwen3_tts_design_then_clone.json`：进阶示例，展示如何先设计声音再进行克隆。
3.  `examples/qwen3_tts_full_studio.json`：**Stage Manager 全功能示例**，支持多角色对话、剧本解析、声音克隆与设计混合编排。


您可以直接将 JSON 文件拖入 ComfyUI 界面即可加载。
