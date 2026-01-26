# Qwen-TTS 灵活角色扮演与剧本导演方案

本文档介绍了如何在 ComfyUI 中利用 Qwen-TTS 的 **Voice Design (声音设计)** 功能，实现灵活的角色扮演和剧本生成。

## 核心概念

核心思想是将 **身份 (Identity)** 与 **表演 (Performance)** 分离：

-   **声音描述 (Voice Description)**：角色的永久特征（如：年龄、性别、音色）。例如：“一个声音低沉沙哑的老巫师”。
-   **情感/风格 (Style/Emotion)**：当下说话的方式（如：耳语、大喊、大笑）。例如：“愤怒地大叫”。

将这两者结合，可以让同一个“角色”演绎出丰富多变的情感。

## 新增工具

### 1. `Qwen3TTS Prompt Composer` (提示词合成器)

这是一个辅助节点，用于简化提示词的拼接。

*   **输入**:
    *   `voice_description`: 定义角色的声音特征。
    *   `style_emotion`: 定义当下的情感状态。
*   **输出**:
    *   `instruct`: 组合好的提示词，直接连到 Voice Design 节点。

### 2. 角色扮演示例工作流

请查看 `examples/qwen3_tts_roleplay_demo.json`。

---

## 剧本“导演”模式 (Script Director Mode)

如果你更喜欢直接写剧本，而不是连接很多节点，请使用 **`Qwen3TTS Stage Manager` (舞台管理器)** 节点。

**如何工作：**
1.  **定义角色**：为角色 A、B、C 分配名字（如“旁白”、“英雄”）和声音描述。
2.  **编写剧本**：在脚本框中粘贴对话。

**剧本格式：**
1.  **无限角色定义 (Casting Sheet)**：
    请在节点上的 `role_definitions` 文本框中定义角色（一行一个），不需要 `def`。
    ```
    国王: 威严的声音，低沉
    士兵: 有点紧张的声音
    ```
2.  **对话内容 (Script)**：
    在 `script` 文本框中只写对话：
    ```
    角色名：(情感) 台词内容...
    ```
*(支持中文冒号 `：` 和括号 `（）`)*

**多角色剧本示例：**
*Role Definitions:*
```
Wizard: A wise old man, raspy voice.
Warrior: A strong female voice.
```
*Script:*
```
Wizard: (Mysterious) The seal is broken.
Warrior: (Shocked) What do we do now?
Narrator: (Calm) And so the journey begins.
法师：(焦急) 快跑！
```

**输出功能：**
该节点会自动根据剧本切换声音和情感，并输出：
1.  **Audio Mix (混合音频)**：完整的场景音频。
2.  **Audio Roles A/B/C (分轨音频)**：前3个角色的独立音轨。
3.  **SRT Content (字幕内容)**：包含精确时间轴的 SRT 字幕文本。
4.  **Save to File (保存文件)**：如果开启此选项，**所有角色**（哪怕有20个）的分轨音频都会自动保存到 ComfyUI 的输出目录。

请查看 `examples/qwen3_tts_full_studio.json` 获取完整示例。

## **进阶功能：声音克隆 (Voice Cloning)**

如果你想使用特定的声音（如你自己的录音），而不是由 AI 捏造声音：

1.  **连接克隆模型**：
    Stage Manager 节点新增了 `clone_model` 接口。**必须**连接 `Qwen3-TTS-Base`（不是 VoiceDesign）模型及 Text 提示词模型。

2.  **连接音频文件**：
    节点新增了 7 个音频输入口：`role_A_audio` ~ `role_G_audio`。
    使用 `LoadAudio` 节点上传你的 wav/mp3 文件，并连接到对应的插口。

3.  **自动激活**：
    只要对应的角色连接了音频，该角色会自动切换为 **克隆模式 (Clone Mode)**。
    
    *   **角色 A, B, C**: 直接对应 `role_A_audio`, `role_B_audio`, `role_C_audio`。
    *   **角色 D, E, F...**: 在 Casting Sheet 中新定义的角色，会按顺序自动使用剩余的音频插口 (D, E, F...)。

**混用示例**：
你可以在同一个剧本里，让“国王”（AI捏造，Design模式）和“王后”（必须是你上传的录音，Clone模式）进行对话。只需给王后连上音频线即可。

## 提示词技巧

*   **具体一点**：“一个男人”效果很差。“一个50多岁、抽烟嗓的侦探”效果很好。
*   **使用形容词**：沙哑的 (gravelly)、温柔的 (gentle/smooth)、尖锐的 (high-pitched)、结巴的 (stuttering)。
*   **利用上下文**：Qwen-TTS 理解语境。“像海盗一样说话”也是可以的！

**专家技巧：声音注入 (Voice Injection)**

为了避免盲目尝试，推荐使用以下工作流：
1.  **先捏 (Preview)**：使用独立的 `Qwen3TTS Voice Design` 节点，输入形容词（如“威严的声音”），反复生成直到满意。
2.  **再用 (Inject)**：将满意的音频输出连接到 `StageManager` 的 `role_A_audio`。
3.  `StageManager` 会以这段音频为蓝本，克隆出完全一致的角色声音。

祝你在虚拟片场玩得开心！
