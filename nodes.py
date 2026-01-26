
import os
import torch
import numpy as np
import folder_paths
from qwen_tts import Qwen3TTSModel
import re
import torchaudio

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
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            }
        }
    
    RETURN_TYPES = ("QWEN3_TTS_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3TTS"

    def load_model(self, model_id, precision, device="auto"):
        dtype = torch.float32
        if precision == "bf16":
            dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16

        print(f"Loading Qwen3-TTS model: {model_id} with precision {precision} on device {device}")
        
        local_path = os.path.join(folder_paths.models_dir, "qwen3_tts", model_id.split("/")[-1])
        model_name_or_path = model_id
        if os.path.exists(local_path):
             model_name_or_path = local_path
             print(f"Found local model at: {local_path}")
        
        # Determine attention implementation
        attn_impl = "sdpa" # Default safe fallback
        if torch.cuda.is_available() and (device == "auto" or device == "cuda"):
            try:
                import flash_attn
                attn_impl = "flash_attention_2"
            except ImportError:
                pass

        model = Qwen3TTSModel.from_pretrained(
            model_name_or_path,
            device_map=device,
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, speaker, language, instruct="", top_p=1.0, temperature=0.9, repetition_penalty=1.05, max_new_tokens=2048, top_k=50, do_sample=True, enable_text_normalization=True, seed=0):
        if model.model.tts_model_type != "custom_voice":
             raise ValueError(f"Loaded model is type '{model.model.tts_model_type}', but 'custom_voice' is required for this node.")

        if seed is not None:
            torch.manual_seed(seed)

        target_lang = None if language == "Auto" else language
        
        print(f"Generating CustomVoice: {text[:50]}...")
        
        wavs, output_sr = model.generate_custom_voice(
            text=text,
            language=target_lang,
            speaker=speaker,
            instruct=instruct if instruct else None,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            do_sample=do_sample,
            enable_text_normalization=enable_text_normalization,
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, text, instruct, language, top_p=1.0, temperature=0.9, repetition_penalty=1.05, max_new_tokens=2048, top_k=50, do_sample=True, enable_text_normalization=True, seed=0):
        if model.model.tts_model_type != "voice_design":
             raise ValueError(f"Loaded model is type '{model.model.tts_model_type}', but 'voice_design' is required for this node.")

        if seed is not None:
            torch.manual_seed(seed)

        target_lang = None if language == "Auto" else language
        
        print(f"Generating VoiceDesign: {text[:50]}...")
        
        wavs, output_sr = model.generate_voice_design(
            text=text,
            language=target_lang,
            instruct=instruct,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            do_sample=do_sample,
            enable_text_normalization=enable_text_normalization,
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3TTS"

    def generate(self, model, ref_audio, text, language, ref_text="", x_vector_only=False, top_p=1.0, temperature=0.9, repetition_penalty=1.05, max_new_tokens=2048, top_k=50, do_sample=True, enable_text_normalization=True, seed=0):
        if model.model.tts_model_type != "base":
             raise ValueError(f"Loaded model is type '{model.model.tts_model_type}', but 'base' is required for Voice Clone.")

        if seed is not None:
             torch.manual_seed(seed)

        target_lang = None if language == "Auto" else language
        
        # Prepare reference audio from AUDIO input
        # ComfyUI AUDIO: {"waveform": tensor [batch, channels, samples], "sample_rate": int}
        waveform = ref_audio["waveform"]
        sr = ref_audio["sample_rate"]
        print(f"DEBUG: VoiceClone input waveform shape: {waveform.shape}, sr: {sr}")
        w_np = waveform.cpu().numpy()
        if w_np.ndim == 3: w_np = w_np[0] 
        
        # w_np is now likely [channels, samples] (e.g. [2, 272319]) or [samples, channels]
        # Heuristic: Audio is usually longer than it is wide (channels)
        if w_np.ndim == 2:
            if w_np.shape[0] < w_np.shape[1]: 
                # Shape is [channels, samples], e.g. [2, 48000]
                # transpose to [samples, channels] for consistency or just average axis 0
                w_np = np.mean(w_np, axis=0)
            else:
                # Shape is [samples, channels], e.g. [48000, 2]
                w_np = np.mean(w_np, axis=1)
        
        # Ensure it's 1D array for mono
        if w_np.ndim > 1:
            w_np = w_np.flatten()
            
        # Check for minimum length (e.g. 0.5 seconds worth of samples, or at least enough for padding)
        # Pad 384 means at least that many samples.
        if w_np.shape[0] < 400: # Very permissive minimum
            raise ValueError(f"Reference audio is too short! Got {w_np.shape[0]} samples. Please provide a longer audio clip (at least 1 second recommended).")

            
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
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            do_sample=do_sample,
            enable_text_normalization=enable_text_normalization,
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

class Qwen3TTSRefAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "text": ("STRING", {"multiline": True, "default": "The transcript of the audio."}),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "wrap_audio"
    CATEGORY = "Qwen3TTS"

    def wrap_audio(self, audio, text):
        # Create a shallow copy to avoid mutating the original dict if it's reused
        new_audio = audio.copy()
        new_audio["text"] = text
        return (new_audio,)

class Qwen3TTSStageManager:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": "Narrator: (Calm) The adventure begins.\nHero: (Bold) Let's go!"}),
                "role_definitions": ("STRING", {"multiline": True, "default": "Narrator [A]: A clear, neutral voice.\nHero [B]: A brave, young voice."}),
                "my_turn_interval": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
            },
            "optional": {
                "model": ("QWEN3_TTS_MODEL",),
                "clone_model": ("QWEN3_TTS_MODEL",),
                "save_to_file": ("BOOLEAN", {"default": False, "label_on": "Save Tracks"}),
                "filename_prefix": ("STRING", {"default": "stage_manager"}),
            },
            "optional": {
                "role_A_audio": ("AUDIO",),
                "role_B_audio": ("AUDIO",),
                "role_C_audio": ("AUDIO",),
                "role_D_audio": ("AUDIO",),
                "role_E_audio": ("AUDIO",),
                "role_F_audio": ("AUDIO",),
                "role_G_audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING",)
    RETURN_NAMES = ("audio_mix", "audio_role_A/0", "audio_role_B/1", "audio_role_C/2", "srt_content",)
    FUNCTION = "generate_scene"
    CATEGORY = "Qwen3TTS"

    def generate_scene(self, script, role_definitions, my_turn_interval=0.5, 
                       model=None, clone_model=None,
                       save_to_file=False, filename_prefix="stage_manager", 
                       role_A_audio=None, role_B_audio=None, role_C_audio=None, 
                       role_D_audio=None, role_E_audio=None, role_F_audio=None, role_G_audio=None):
        
        # Validation checks
        if model is None and clone_model is None:
             raise ValueError("Stage Manager needs at least one model! Connect either 'model' (for creation) or 'clone_model' (for cloning).")

        if model is not None and model.model.tts_model_type != "voice_design":
             # Warn but don't crash? Or crashing is safer.
             raise ValueError(f"Stage Manager 'model' input requires a 'voice_design' model, but loaded '{model.model.tts_model_type}'.")
        
        has_clone_model = False
        if clone_model is not None:
             has_clone_model = True

        # Config map
        roles_config = {}
        
        # Audio Slot Map
        audio_slots = {
            "A": role_A_audio, "B": role_B_audio, "C": role_C_audio,
            "D": role_D_audio, "E": role_E_audio, "F": role_F_audio, "G": role_G_audio
        }
        used_slots = set()
        slot_keys = sorted(list(audio_slots.keys())) # A, B, C, D, E, F, G

        # Parse Role Definitions
        def_lines = role_definitions.strip().split('\n')
        
        # Regex: Name [A]: Desc or Name: Desc
        # Group 1: Name, Group 2: Slot (Optional), Group 3: Desc
        def_pattern = re.compile(r"^(.+?)(?:\s*\[([A-G])\])?\s*[:：]\s*(.+)$", re.IGNORECASE)
        
        parsed_roles = [] # list of (name, requested_slot, desc)

        for line in def_lines:
            line = line.strip()
            if not line: continue
            
            match = def_pattern.match(line)
            if match:
                r_name = match.group(1).strip()
                r_slot = match.group(2).upper() if match.group(2) else None
                r_desc = match.group(3).strip()
                parsed_roles.append({'name': r_name, 'req': r_slot, 'desc': r_desc})
                if r_slot:
                    used_slots.add(r_slot)
        
        # Assignment Logic
        for role in parsed_roles:
            r_name = role['name']
            r_req = role['req']
            r_desc = role['desc']
            
            final_slot = None
            
            if r_req:
                # Explicit assignment
                final_slot = r_req
            else:
                # Auto-assignment: Find first unused slot
                for s in slot_keys:
                    if s not in used_slots:
                        final_slot = s
                        used_slots.add(s)
                        break
            
            # Determine Audio Input
            role_audio_input = None
            if final_slot:
                role_audio_input = audio_slots.get(final_slot)
            
            # File Check
            possible_path = r_desc.strip('"').strip("'")
            is_file = os.path.isfile(possible_path)
            
            roles_config[r_name] = {
                "desc": r_desc, 
                "static_id": final_slot if final_slot in ["A", "B", "C"] else None, 
                "audio_input": role_audio_input, 
                "is_file": is_file
            }
            
            current_audio = f"Input {final_slot}" if role_audio_input is not None else ("File" if is_file else "None")
            print(f"Role Configured: {r_name} -> Slot [{final_slot or 'None'}] -> {current_audio}")

        # 3. Parse Script
        script_lines = script.strip().split('\n')
        print(f"DEBUG: Parsed {len(script_lines)} lines from script.")

        # Dynamic Timelines
        timelines = {}
        for r_name in roles_config:
            timelines[r_name] = []
            
        sample_rate = 24000 
        srt_lines = []
        current_time_seconds = 0.0
        
        pattern = re.compile(r"^(.+?)[:：](?:\s*[(\uff08](.+?)[)\uff09])?\s*(.+)$")

        def format_timestamp(seconds):
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            
        def get_tensor_props(t):
            return t.shape[1], t.shape[2] 

        valid_line_count = 0
        
        # Prepare Audio for Cloning
        def process_ref_audio(audio_data):
            # Returns tuple (numpy_wave, sr) suitable for Qwen
            # audio_data is users dictionary {"waveform":..., "sample_rate":...}
            waveform = audio_data["waveform"]
            sr = audio_data["sample_rate"]
            w_np = waveform.cpu().numpy()
            if w_np.ndim == 3: w_np = w_np[0]
            if w_np.ndim == 2:
                if w_np.shape[0] < w_np.shape[1]: w_np = np.mean(w_np, axis=0)
                else: w_np = np.mean(w_np, axis=1)
            if w_np.ndim > 1: w_np = w_np.flatten()
            return (w_np, sr)

        for i, line in enumerate(script_lines):
            print(f"DEBUG: Processing Line {i+1}/{len(script_lines)}: {line[:30]}...")
            line = line.strip()
            if not line: continue
            
            match = pattern.match(line)
            if not match: 
                print(f"DEBUG: Regex failed match for line: {line}")
                continue
                
            role_name = match.group(1).strip()
            emotion = match.group(2)
            content = match.group(3).strip()
            
            active_role_key = None
            for r_key in roles_config:
                if r_key.lower() == role_name.lower():
                    active_role_key = r_key
                    break
            
            if not active_role_key:
                print(f"Role '{role_name}' not defined. Skipping.")
                continue
                
            role_data = roles_config[active_role_key]
            
            # Determine Mode: Clone vs Design
            # Clone if: Audio Input Present OR Description is File Path
            is_clone_mode = False
            ref_audio_obj = None
            
            if role_data.get("audio_input") is not None:
                is_clone_mode = True
                audio_input_data = role_data["audio_input"]
                ref_audio_obj = process_ref_audio(audio_input_data)
                
                # Check for bundled text (from Qwen3TTSRefAudio node)
                if "text" in audio_input_data and audio_input_data["text"]:
                    ref_text_obj = audio_input_data["text"]
                    x_vector = False
                    print(f"Role {role_name}: Found ref_text in Audio input. Using ICL Mode.")
                else:
                    ref_text_obj = None
                    x_vector = True
                    print(f"Role {role_name}: No ref_text in Audio input. Using X-Vector Mode.")
                    
            elif role_data.get("is_file", False):
                is_clone_mode = True
                # Load from file
                fpath = role_data["desc"].strip('"').strip("'")
                try:
                    w, sr = torchaudio.load(fpath)
                    # Convert to Qwen format
                    w_np = w.numpy()
                    if w_np.ndim == 2: w_np = np.mean(w_np, axis=0)
                    ref_audio_obj = (w_np, sr)
                    
                    # File loading doesn't support text yet
                    ref_text_obj = None
                    x_vector = True 

                except Exception as e:
                    print(f"Failed to load audio file {fpath}: {e}. Falling back to Design.")
                    is_clone_mode = False
            
            if is_clone_mode and not has_clone_model:
                print(f"Role {role_name} requested cloning but no 'clone_model' connected. Falling back to Design.")
                is_clone_mode = False
                
            # Generation
            wavs = []
            output_sr = 24000
            
            if is_clone_mode:
                print(f"Generating Line {valid_line_count+1} [CLONE]: {role_name}")
                wavs, output_sr = clone_model.generate_voice_clone(
                    text=content,
                    language="Auto",
                    ref_audio=ref_audio_obj,
                    ref_text=ref_text_obj,
                    x_vector_only_mode=x_vector,
                    do_sample=True
                )
            else:
                # Design Mode
                if model is None:
                    raise ValueError(f"Role '{role_name}' requires Voice Creation (Design Mode), but no 'model' (VoiceDesign) is connected to the StageManager. Please connect a Qwen3-VoiceDesign model or provide audio input for this role.")

                print(f"Generating Line {valid_line_count+1} [DESIGN]: {role_name}")
                voice_desc = role_data["desc"]
                if emotion: instruct = f"{emotion.strip()}, {voice_desc}"
                else: instruct = voice_desc
                
                wavs, output_sr = model.generate_voice_design(
                    text=content,
                    language="Auto",
                    instruct=instruct,
                    do_sample=True
                )
            
            if not wavs: continue
            
            sample_rate = output_sr
            
            # Process Output
            w = wavs[0]
            t = torch.from_numpy(w)
            if t.ndim == 1: t = t.unsqueeze(0)
            elif t.ndim == 2 and t.shape[0] > t.shape[1]: t = t.t()
            t = t.unsqueeze(0) # [1, C, L]
            
            channels, audio_samples = get_tensor_props(t)
            duration_seconds = audio_samples / sample_rate
            
            # Update SRT
            valid_line_count += 1
            start_time = current_time_seconds
            end_time = start_time + duration_seconds
            srt_lines.append(f"{valid_line_count}\n{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n{role_name}: {content}\n")
            
            # Update Timelines
            # Add audio to active role, silence to ALL others
            for r_key, tl in timelines.items():
                if r_key == active_role_key:
                    tl.append(t)
                else:
                    silence = torch.zeros((1, channels, audio_samples), dtype=torch.float32)
                    tl.append(silence)

            current_time_seconds += duration_seconds
            
            # Interval Silence
            if my_turn_interval > 0:
                silence_samples = int(my_turn_interval * sample_rate)
                silence_gap = torch.zeros((1, channels, silence_samples), dtype=torch.float32)
                for tl in timelines.values():
                    tl.append(silence_gap)
                current_time_seconds += my_turn_interval

        # Finalize Outputs
        # Cat timelines
        raw_outputs = {} # role_key -> tensor full
        
        # Max length align
        max_len = 0
        for r_key, tl in timelines.items():
            if not tl:
                 # If a defined role never spoke, create dummy silence of 0.1s
                 dummy = torch.zeros((1, 1, int(sample_rate * 0.1)))
                 raw_outputs[r_key] = dummy
            else:
                 cat_t = torch.cat(tl, dim=2)
                 raw_outputs[r_key] = cat_t
                 if cat_t.shape[2] > max_len: max_len = cat_t.shape[2]

        # Pad to max_len
        final_outputs = {}
        for r_key, t in raw_outputs.items():
            current_len = t.shape[2]
            if current_len < max_len:
                padding = torch.zeros((t.shape[0], t.shape[1], max_len - current_len), dtype=t.dtype)
                t = torch.cat([t, padding], dim=2)
            final_outputs[r_key] = t
            
        # Mix
        final_mix = torch.zeros((1, 1, max_len)) # Init accumulator
        for t in final_outputs.values():
            if t.shape[1] > final_mix.shape[1]: 
                final_mix = final_mix.repeat(1, t.shape[1], 1)
            elif final_mix.shape[1] > t.shape[1]:
                t = t.repeat(1, final_mix.shape[1], 1)
            final_mix = final_mix + t

        # Map to Static Outputs A/B/C
        out_A = torch.zeros((1, 1, max_len))
        out_B = torch.zeros((1, 1, max_len))
        out_C = torch.zeros((1, 1, max_len))
        
        for r_key, r_data in roles_config.items():
            t = final_outputs.get(r_key)
            if t is None: continue
            
            sid = r_data["static_id"]
            if sid == "A": out_A = t
            elif sid == "B": out_B = t
            elif sid == "C": out_C = t
            
        # Save to File if requested
        if save_to_file:
            output_dir = folder_paths.get_output_directory()
            # Save Mix
            torchaudio.save(os.path.join(output_dir, f"{filename_prefix}_MIX.wav"), final_mix[0], sample_rate)
            # Save Roles
            for r_key, t in final_outputs.items():
                safe_name = "".join([c for c in r_key if c.isalnum() or c in (' ', '_', '-')]).strip()
                fname = f"{filename_prefix}_{safe_name}.wav"
                torchaudio.save(os.path.join(output_dir, fname), t[0], sample_rate)
            print(f"Saved tracks to {output_dir}")

        srt_output = "\n".join(srt_lines)
        
        return (
            {"waveform": final_mix, "sample_rate": sample_rate},
            {"waveform": out_A, "sample_rate": sample_rate},
            {"waveform": out_B, "sample_rate": sample_rate},
            {"waveform": out_C, "sample_rate": sample_rate},
            srt_output,
        )

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSStageManager": Qwen3TTSStageManager,
    "Qwen3TTSRefAudio": Qwen3TTSRefAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3TTSLoader": "Qwen3 TTS Model Loader",
    "Qwen3TTSCustomVoice": "Qwen3 TTS Custom Voice",
    "Qwen3TTSVoiceDesign": "Qwen3 TTS Voice Design",
    "Qwen3TTSVoiceClone": "Qwen3 TTS Voice Clone",
    "Qwen3TTSStageManager": "Qwen3 TTS Stage Manager 🎬",
    "Qwen3TTSRefAudio": "Qwen3 TTS Ref Audio (Audio+Text)",
}
