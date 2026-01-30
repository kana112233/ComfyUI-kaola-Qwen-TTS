
import os
import torch
import numpy as np
import folder_paths
from qwen_tts import Qwen3TTSModel
import re
import torchaudio
import soundfile as sf
import comfy.utils

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
                import importlib
                # Check directly if version metadata is available, as transformers will error otherwise
                importlib.metadata.version("flash_attn")
                attn_impl = "flash_attention_2"
            except Exception:
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
                "model": ("QWEN3_TTS_MODEL",),
                "script": ("STRING", {"multiline": True, "default": "Narrator: The adventure begins.\nHero: Let's go!"}),
                "role_definitions": ("STRING", {"multiline": True, "default": "Narrator [A]: A clear, neutral voice.\nHero [B]: A brave, young voice."}),
                "my_turn_interval": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1}),
                "overlap_handling": (["ignore", "shift_start", "truncate"], {"default": "ignore"}),
            },
            "optional": {
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "save_to_file": ("BOOLEAN", {"default": False, "label_on": "Save Tracks"}),
                "filename_prefix": ("STRING", {"default": "stage_manager"}),
                "role_A_audio": ("AUDIO",),
                "role_B_audio": ("AUDIO",),
                "role_C_audio": ("AUDIO",),
                "role_D_audio": ("AUDIO",),
                "role_E_audio": ("AUDIO",),
                "role_F_audio": ("AUDIO",),
                "role_G_audio": ("AUDIO",),
                "language": (
                    ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
                    {"default": "Auto"}
                ),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 100}),
                "enable_text_normalization": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING",)
    RETURN_NAMES = ("audio_mix", "audio_role_A", "audio_role_B", "audio_role_C", "audio_role_D", "audio_role_E", "audio_role_F", "audio_role_G", "srt_content",)
    FUNCTION = "generate_scene"
    CATEGORY = "Qwen3TTS"

    def generate_scene(self, model, script, role_definitions, my_turn_interval=0.5, 
                       top_p=1.0, temperature=0.7, repetition_penalty=1.05, seed=0,
                       save_to_file=False, filename_prefix="stage_manager", 
                       role_A_audio=None, role_B_audio=None, role_C_audio=None, 
                       role_D_audio=None, role_E_audio=None, role_F_audio=None, role_G_audio=None,
                       language="Auto", max_new_tokens=2048, top_k=50, enable_text_normalization=True,
                       overlap_handling="ignore"):
        
        # Validation checks
        if model is None:
             raise ValueError("Stage Manager needs a 'model'!")
        
        # Unified Model Mode: Use the single provided model for everything
        voice_clone_worker = model

        model_type = model.model.tts_model_type
        print(f"StageManager loaded model type: {model_type}")
        
        # Config map
        roles_config = {}
        
        # Audio Slot Map
        audio_slots = {
            "A": role_A_audio, "B": role_B_audio, "C": role_C_audio,
            "D": role_D_audio, "E": role_E_audio, "F": role_F_audio, "G": role_G_audio
        }
        for k, v in audio_slots.items():
            if v is not None:
                print(f"DEBUG: StageManager received input for Slot {k}")

        used_slots = set()
        slot_keys = sorted(list(audio_slots.keys())) # A, B, C, D, E, F, G

        # Parse Role Definitions
        print(f"DEBUG: Parsing Role Definitions:\n{role_definitions}")
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
                "static_id": final_slot if final_slot in ["A", "B", "C", "D", "E", "F", "G"] else None, 
                "audio_input": role_audio_input, 
                "is_file": is_file
            }
            
            current_audio = f"Input {final_slot}" if role_audio_input is not None else ("File" if is_file else "None")
            print(f"Role Configured: {r_name} -> Slot [{final_slot or 'None'}] -> {current_audio}")

        # 3. Parse Script
        script_lines = script.strip().split('\n')
        print(f"DEBUG: Parsed {len(script_lines)} lines from script.")

        # --- Pre-scan for missing roles ---
        found_script_roles = set()
        for line in script_lines:
            line = line.strip()
            if not line: continue
            
            # Re-use regex patterns defined later or define them here temporarily? 
            # Better to lift regex definitions up or just duplicate/use local ones for this scan.
            # Using simple check consistent with main loop logic.
            # Main loop uses: ts_pattern and pattern. Let's move them up.
            pass # Placeholder to just insert the block below correctly
        
        # Regex definitions moved up for pre-scan
        ts_pattern = re.compile(r"^(\d+)\s+([0-9:,\.]+)\s*-{2,}>\s*([0-9:,\.]+)\s+(.+?)[:：]\s*(.+)$")
        pattern = re.compile(r"^([^0-9\s].*?)[:：]\s*(.+)$")

        for line in script_lines:
            line = line.strip()
            if not line: continue
            
            m_ts = ts_pattern.match(line)
            m_std = pattern.match(line)
            
            r_found = None
            if m_ts:
                r_found = m_ts.group(4).strip()
            elif m_std:
                r_found = m_std.group(1).strip()
            
            if r_found:
                found_script_roles.add(r_found)
        
        defined_roles_lower = {k.lower(): k for k in roles_config.keys()}
        missing_roles = []
        
        for r in found_script_roles:
            if r.lower() not in defined_roles_lower:
                missing_roles.append(r)
        
        if missing_roles:
            err_msg = "The following roles appear in the script but are NOT defined in Role Definitions:\n"
            for mr in missing_roles:
                err_msg += f"  - '{mr}'\n"
            err_msg += "Please add them to the 'Role Definitions' text box (e.g. speakerUnknown [A]: description)"
            raise ValueError(f"StageManager Error:\n{err_msg}")
        else:
            print(f"DEBUG: All script roles are defined: {list(found_script_roles)}")
        # ----------------------------------

        sample_rate = 24000 
        srt_lines = []
        
        # Absolute Timing Layout Engine
        # events: list of match objects or dicts {start_time, end_time (optional), role, content}
        timeline_events = [] 
        
        # Regex 1: Explicit Timestamp + Speaker format
        # Relaxed pattern to allow flexible spacing
        ts_pattern = re.compile(r"^(\d+)\s+([0-9:,\.]+)\s*-{2,}>\s*([0-9:,\.]+)\s+(.+?)[:：]\s*(.+)$")

        # Regex 2: Standard Name: Content
        # Avoid matching timestamp partials by requiring name to not be just digits/timestamps
        pattern = re.compile(r"^([^0-9\s].*?)[:：]\s*(.+)$")
        
        def parse_timestamp_to_seconds(ts_str):
            # Format: HH:MM:SS,mmm
            try:
                time_part, millis_part = ts_str.split(',')
                h, m, s = map(int, time_part.split(':'))
                millis = int(millis_part)
                return h * 3600 + m * 60 + s + millis / 1000.0
            except:
                return 0.0

        def format_timestamp(seconds):
            millis = int((seconds % 1) * 1000)
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
            
        def get_tensor_props(t):
            return t.shape[1], t.shape[2] 
        
        def process_ref_audio(audio_data):
            # Returns tuple (numpy_wave, sr) suitable for Qwen
            waveform = audio_data["waveform"]
            sr = audio_data["sample_rate"]
            w_np = waveform.cpu().numpy()
            if w_np.ndim == 3: w_np = w_np[0]
            if w_np.ndim == 2:
                if w_np.shape[0] < w_np.shape[1]: w_np = np.mean(w_np, axis=0)
                else: w_np = np.mean(w_np, axis=1)
            if w_np.ndim > 1: w_np = w_np.flatten()
            
            if w_np.shape[0] < 400:
                print(f"ERROR: Reference audio for {audio_data.get('role', 'unknown')} is too short: {w_np.shape[0]} samples.")
                raise ValueError(f"StageManager: Reference audio is too short (<400 samples).")

            return (w_np, sr)

        cursor_time = 0.0 # Tracks the end of the last generate clip for auto-placement
        
        # Collection Phase
        valid_line_count = 0
        last_valid_role = None

        pbar = comfy.utils.ProgressBar(len(script_lines))
        for i, line in enumerate(script_lines):
            pbar.update(1)
            
            if seed is not None:
                line_seed = seed + i
                torch.manual_seed(line_seed)
            
            line = line.strip()
            if not line: continue
            
            match_ts = ts_pattern.match(line)
            match_std = pattern.match(line)
            
            role_name = None
            content = None
            explicit_start_time = None
            explicit_end_time = None
            
            if match_ts:
                explicit_start_time = parse_timestamp_to_seconds(match_ts.group(2))
                explicit_end_time = parse_timestamp_to_seconds(match_ts.group(3))
                role_name = match_ts.group(4).strip()
                content = match_ts.group(5).strip()
                last_valid_role = role_name
                print(f"Line {i}: Found explicit timestamp {match_ts.group(2)} ({explicit_start_time}s) -> {match_ts.group(3)} ({explicit_end_time}s)")
                
            elif match_std:
                role_name = match_std.group(1).strip()
                content = match_std.group(2).strip()
                last_valid_role = role_name
            else:
                if last_valid_role:
                    role_name = last_valid_role
                    content = line
                else: 
                    continue
                
            active_role_key = None
            for r_key in roles_config:
                if r_key.lower() == role_name.lower():
                    active_role_key = r_key
                    break
            
            if not active_role_key:
                print(f"Role '{role_name}' not defined. Skipping.")
                continue
                
            role_data = roles_config[active_role_key]
            
            # Determine Mode
            is_clone_mode = False
            ref_audio_obj = None
            ref_text_obj = None
            x_vector = True
            target_lang = None if language == "Auto" else language
            
            if role_data.get("audio_input") is not None:
                is_clone_mode = True
                audio_input_data = role_data["audio_input"]
                ref_audio_obj = process_ref_audio(audio_input_data)
                print(f"DEBUG: Role {active_role_key} - Using Input Audio. Ref shape: {ref_audio_obj[0].shape}, SR: {ref_audio_obj[1]}")
                if "text" in audio_input_data and audio_input_data["text"]:
                    ref_text_obj = audio_input_data["text"]
                    x_vector = False
            elif role_data.get("is_file", False):
                is_clone_mode = True
                fpath = role_data["desc"].strip('"').strip("'")
                try:
                    w, sr = torchaudio.load(fpath)
                    w_np = w.numpy()
                    if w_np.ndim == 2: w_np = np.mean(w_np, axis=0)
                    ref_audio_obj = (w_np, sr)
                except Exception as e:
                    print(f"Failed to load audio file {fpath}: {e}. Falling back to Design.")
                    is_clone_mode = False
            
            # Generate Audio
            print(f"DEBUG: Generating line {valid_line_count+1} for {role_name} (Clone={is_clone_mode})...")
            wavs = []
            output_sr = 24000
            
            if is_clone_mode:
                wavs, output_sr = voice_clone_worker.generate_voice_clone(
                    text=content,
                    language=target_lang,
                    ref_audio=ref_audio_obj,
                    ref_text=ref_text_obj,
                    x_vector_only_mode=x_vector,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    enable_text_normalization=enable_text_normalization,
                )
            else:
                instruct = role_data["desc"]
                wavs, output_sr = model.generate_voice_design(
                    text=content,
                    language=target_lang,
                    instruct=instruct,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    top_k=top_k,
                    enable_text_normalization=enable_text_normalization,
                )
            
            if not wavs: 
                print(f"DEBUG: Generation failed/empty for line {i}. Content='{content}'.")
                print("       (Note: TTS models often fail on text that is purely in brackets [] or consist only of sound effects.)")
                continue
            print(f"DEBUG: Generated {len(wavs)} wav segments. Generation finished for line {valid_line_count+1}.")
            sample_rate = output_sr
            
            # Convert to Tensor
            w = wavs[0]
            t = torch.from_numpy(w)
            if t.ndim == 1: t = t.unsqueeze(0)
            elif t.ndim == 2 and t.shape[0] > t.shape[1]: t = t.t()
            t = t.unsqueeze(0) # [1, C, L]
            
            channels, audio_samples = get_tensor_props(t)
            duration_seconds = audio_samples / sample_rate
            
            # Determine Placement
            final_start_time = 0.0
            
            if explicit_start_time is not None:
                final_start_time = explicit_start_time
                
                if final_start_time < cursor_time:
                    if overlap_handling == "shift_start":
                        # Delay this line to start after previous one finishes
                        print(f"DEBUG: Shifted start from {final_start_time:.3f} to {cursor_time:.3f} to avoid overlap.")
                        final_start_time = cursor_time
                    elif overlap_handling == "truncate":
                        # Truncate logic handled below using explicit_end_time if available
                        pass

                # Update cursor to end of this clip for next implicit line
                cursor_time = final_start_time + duration_seconds
                
                # Truncate Logic: Enforce explicit end time if provided
                if overlap_handling == "truncate" and explicit_end_time is not None:
                     allowed_duration = explicit_end_time - final_start_time
                     if duration_seconds > allowed_duration:
                         print(f"DEBUG: Truncating audio for line {i}: {duration_seconds:.3f}s -> {allowed_duration:.3f}s")
                         # Slice the tensor
                         max_samples = int(allowed_duration * sample_rate)
                         if max_samples < t.shape[2]:
                             t = t[:, :, :max_samples]
                             # Update duration and cursor
                             duration_seconds = allowed_duration
                             cursor_time = final_start_time + duration_seconds
            else:
                # Implicit placement: use cursor (previous end + interval)
                # Apply interval before starting this implicit line
                final_start_time = cursor_time + (my_turn_interval if valid_line_count > 0 else 0)
                cursor_time = final_start_time + duration_seconds

            valid_line_count += 1
            start_time_fmt = format_timestamp(final_start_time)
            end_time_fmt = format_timestamp(final_start_time + duration_seconds)
            srt_lines.append(f"{valid_line_count}\n{start_time_fmt} --> {end_time_fmt}\n{role_name}: {content}\n")
            
            # Store Event
            timeline_events.append({
                "role": active_role_key,
                "start_time": final_start_time,
                "end_time": final_start_time + duration_seconds,
                "tensor": t,
                "channels": channels
            })

        # Layout Phase
        # 1. Determine Total Duration & Max Channels
        total_duration = 0.0
        max_channels = 1
        for evt in timeline_events:
            if evt["end_time"] > total_duration:
                total_duration = evt["end_time"]
            if evt["channels"] > max_channels:
                max_channels = evt["channels"]
        
        # Add a tiny buffer at end
        total_duration += 0.5 
        total_samples = int(total_duration * sample_rate)
        
        # 2. Allocate Tracks
        # role_key -> tensor [1, max_channels, total_samples]
        track_buffers = {}
        for r_key in roles_config:
            track_buffers[r_key] = torch.zeros((1, max_channels, total_samples), dtype=torch.float32)
            
        # 3. Paste Events
        for evt in timeline_events:
            r_key = evt["role"]
            t = evt["tensor"]
            start_sec = evt["start_time"]
            
            start_sample = int(start_sec * sample_rate)
            clip_samples = t.shape[2]
            clip_channels = t.shape[1]
            
            # Channel Promotion check
            # t is [1, C, L]
            if clip_channels < max_channels:
                 # Expand mono to stereo/multi
                 t = t.repeat(1, max_channels, 1)
            
            # Boundary check
            end_sample = start_sample + clip_samples
            if end_sample > total_samples:
                end_sample = total_samples
                t = t[:, :, :total_samples - start_sample]
            
            # Add (mix) into buffer to allow overlaps within same role? 
            # Usually same role shouldn't overlap self, but using add handles it gracefully.
            track_buffers[r_key][:, :, start_sample:end_sample] += t
            
        # 4. Final Mix
        final_mix = torch.zeros((1, max_channels, total_samples), dtype=torch.float32)
        for t in track_buffers.values():
            final_mix += t
            
        # Map to Static Outputs
        out_A = torch.zeros((1, max_channels, total_samples))
        out_B = torch.zeros((1, max_channels, total_samples))
        out_C = torch.zeros((1, max_channels, total_samples))
        out_D = torch.zeros((1, max_channels, total_samples))
        out_E = torch.zeros((1, max_channels, total_samples))
        out_F = torch.zeros((1, max_channels, total_samples))
        out_G = torch.zeros((1, max_channels, total_samples))
        
        for r_key, r_data in roles_config.items():
            t = track_buffers.get(r_key)
            if t is None: continue
            
            sid = r_data["static_id"]
            if sid == "A": out_A = t
            elif sid == "B": out_B = t
            elif sid == "C": out_C = t
            elif sid == "D": out_D = t
            elif sid == "E": out_E = t
            elif sid == "F": out_F = t
            elif sid == "G": out_G = t
            
        # Save to File if requested
        if save_to_file:
            output_dir = folder_paths.get_output_directory()
            
            mix_np = final_mix[0].transpose(0, 1).detach().cpu().numpy()
            sf.write(os.path.join(output_dir, f"{filename_prefix}_MIX.wav"), mix_np, sample_rate)
            
            for r_key, t in track_buffers.items():
                safe_name = "".join([c for c in r_key if c.isalnum() or c in (' ', '_', '-')]).strip()
                fname = f"{filename_prefix}_{safe_name}.wav"
                
                role_np = t[0].transpose(0, 1).detach().cpu().numpy()
                sf.write(os.path.join(output_dir, fname), role_np, sample_rate)
            print(f"Saved tracks to {output_dir}")

        srt_output = "\n".join(srt_lines)
        
        return (
            {"waveform": final_mix, "sample_rate": sample_rate},
            {"waveform": out_A, "sample_rate": sample_rate},
            {"waveform": out_B, "sample_rate": sample_rate},
            {"waveform": out_C, "sample_rate": sample_rate},
            {"waveform": out_D, "sample_rate": sample_rate},
            {"waveform": out_E, "sample_rate": sample_rate},
            {"waveform": out_F, "sample_rate": sample_rate},
            {"waveform": out_G, "sample_rate": sample_rate},
            srt_output,
        )


class Qwen3TTSSaveFile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "filename_prefix": ("STRING", {"default": "vibe_output"}),
                "extension": (["srt", "txt", "csv", "json"], {"default": "srt"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_content",)
    FUNCTION = "save_file"
    CATEGORY = "Qwen3TTS"
    
    def save_file(self, text, filename_prefix="vibe_output", extension="srt"):
        output_dir = folder_paths.get_output_directory()
        # Handle filename conflicts
        def get_unique_filename(directory, prefix, ext):
            counter = 1
            filename = f"{prefix}.{ext}"
            while os.path.exists(os.path.join(directory, filename)):
                filename = f"{prefix}_{counter}.{ext}"
                counter += 1
            return filename

        filename = get_unique_filename(output_dir, filename_prefix, extension)
        full_path = os.path.join(output_dir, filename)
        
        print(f"Saving text to {full_path}")
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        return (text,)

NODE_CLASS_MAPPINGS = {
    "Qwen3TTSLoader": Qwen3TTSLoader,
    "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
    "Qwen3TTSVoiceDesign": Qwen3TTSVoiceDesign,
    "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    "Qwen3TTSStageManager": Qwen3TTSStageManager,
    "Qwen3TTSRefAudio": Qwen3TTSRefAudio,
    "Qwen3TTSSaveFile": Qwen3TTSSaveFile,
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
