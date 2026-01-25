
from .nodes import Qwen3TTSLoader, Qwen3TTSCustomVoice, Qwen3TTSVoiceDesign, Qwen3TTSVoiceClone

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSLoader": "Qwen3TTS Loader",
    "Qwen3TTSCustomVoice": "Qwen3TTS Custom Voice",
    "Qwen3TTSVoiceDesign": "Qwen3TTS Voice Design",
    "Qwen3TTSVoiceClone": "Qwen3TTS Voice Clone"
}
