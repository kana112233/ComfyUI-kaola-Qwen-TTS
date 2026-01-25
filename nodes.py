
import os
import torch
import numpy as np
import folder_paths
from qwen_tts import Qwen3TTSModel

# Add Qwen3-TTS related paths
folder_paths.add_model_folder_path("qwen3_tts", os.path.join(folder_paths.models_dir, "qwen3_tts"))

class Qwen3TTSLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_id": (
                    [
                        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    ],
                    {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"}
                ),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }
    
    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3TTS"

    def load_model(self, model_id, precision):
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        print(f"Loading Qwen3-TTS model: {model_id} with precision {precision}")
        
        local_path = os.path.join(folder_paths.models_dir, "qwen3_tts", model_id.split("/")[-1])
        model_name_or_path = model_id
        if os.path.exists(local_path):
             model_name_or_path = local_path
             print(f"Found local model at: {local_path}")
        
        # Determine attention implementation
        attn_impl = "sdpa" # Default safe fallback
        if torch.cuda.is_available():
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                pass

        model = Qwen3TTSModel.from_pretrained(
            model_name_or_path,
            device_map="auto",
            dtype=dtype,
            attn_implementation=attn_impl, 
        )
        
        return (model,)

class Qwen3TTSCustomVoice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, how are you today?"}),
                "speaker": (
                    [
                        "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", # Chinese
                        "Ryan", "Aiden", # English
                        "Ono_Anna", # Japanese
                        "Sohee", # Korean
                    ], 
                    {"default": "Vivian"}
                ),
                "language": (
                    ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                    {"default": "Auto"}
                ),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": "用特别愤怒的语气说"}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, speaker, language, instruct="", top_p=1.0, temperature=0.9, repetition_penalty=1.05):
        if model.model.tts_model_type != "custom_voice":
             raise ValueError(f"Loaded model is type '{model.model.tts_model_type}', but 'custom_voice' is required for this node.")

        target_lang = None if language == "Auto" else language
        
        print(f"Generating CustomVoice: {text[:50]}...")
        
        wavs, output_sr = model.generate_custom_voice(
            text=text,
            language=target_lang,
            speaker=speaker,
            instruct=instruct if instruct else None,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )
        
        return (process_waves_to_audio(wavs, output_sr),)

class Qwen3TTSVoiceDesign:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "text": ("STRING", {"multiline": True, "default": "Hello, how are you today?"}),
                "instruct": ("STRING", {"multiline": True, "default": "A deep, raspy male voice, speaking slowly."}),
                "language": (
                    ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                    {"default": "Auto"}
                ),
            },
            "optional": {
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, instruct, language, top_p=1.0, temperature=0.9, repetition_penalty=1.05):
        if model.model.tts_model_type != "voice_design":
             raise ValueError(f"Loaded model is type '{model.model.tts_model_type}', but 'voice_design' is required for this node.")

        target_lang = None if language == "Auto" else language
        
        print(f"Generating VoiceDesign: {text[:50]}...")
        
        wavs, output_sr = model.generate_voice_design(
            text=text,
            language=target_lang,
            instruct=instruct,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )
        
        return (process_waves_to_audio(wavs, output_sr),)

class Qwen3TTSVoiceClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_TTS_MODEL",),
                "ref_audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True, "default": "Hello, how are you today?"}),
                "language": (
                    ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                    {"default": "Auto"}
                ),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
                "x_vector_only": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, ref_audio, text, language, ref_text="", x_vector_only=False, top_p=1.0, temperature=0.9, repetition_penalty=1.05):
        if model.model.tts_model_type != "base":
             raise ValueError(f"Loaded model is type '{model.model.tts_model_type}', but 'base' is required for Voice Clone.")

        target_lang = None if language == "Auto" else language
        
        # Prepare reference audio from AUDIO input
        # ComfyUI AUDIO: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
        waveform = ref_audio["waveform"]
        sr = ref_audio["sample_rate"]
        w_np = waveform.cpu().numpy()
        if w_np.ndim == 3: w_np = w_np[0] 
        if w_np.shape[0] < w_np.shape[1]: w_np = w_np.transpose() # [samples, channels]
        ref_audio_processed = (w_np, sr)

        print(f"Generating VoiceClone: {text[:50]}...")
        
        wavs, output_sr = model.generate_voice_clone(
            text=text,
            language=target_lang,
            ref_audio=ref_audio_processed,
            ref_text=ref_text if ref_text else None,
            x_vector_only_mode=x_vector_only,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty
        )
        
        return (process_waves_to_audio(wavs, output_sr),)


def process_waves_to_audio(wavs, output_sr):
    # Convert list of numpy arrays to ComfyUI AUDIO dict
    audio_tensors = []
    for w in wavs:
        t = torch.from_numpy(w)
        if t.ndim == 1:
            t = t.unsqueeze(0) # [1, samples] -> Mono
        elif t.ndim == 2 and t.shape[0] > t.shape[1]:
             t = t.t() # [channels, samples]
        
        t = t.unsqueeze(0) # [1, C, L] batch dim
        audio_tensors.append(t)
        
    if len(audio_tensors) > 0:
        final_tensor = torch.cat(audio_tensors, dim=0)
        return {"waveform": final_tensor, "sample_rate": output_sr}
    return None
