from .nodes import Qwen3TTSLoader, Qwen3TTSCustomVoice, Qwen3TTSVoiceDesign, Qwen3TTSVoiceClone, Qwen3TTSStageManager, Qwen3TTSRefAudio, Qwen3TTSSaveFile

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSStageManager": Qwen3TTSStageManager,
    "Qwen3TTSRefAudio": Qwen3TTSRefAudio,
    "Qwen3TTSSaveFile": Qwen3TTSSaveFile
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSLoader": "Model Loader",
    "Qwen3TTSCustomVoice": "Custom Voice (Prompt)",
    "Qwen3TTSVoiceDesign": "Voice Design (Text)",
    "Qwen3TTSVoiceClone": "Voice Clone",
    "Qwen3TTSStageManager": "Stage Manager 🎬",
    "Qwen3TTSRefAudio": "Ref Audio (Audio+Text)",
    "Qwen3TTSSaveFile": "VibeVoice Save File",
}
