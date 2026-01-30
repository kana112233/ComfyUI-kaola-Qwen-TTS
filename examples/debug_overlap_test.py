import sys
import os
import torch
import numpy as np
import types

# --- MOCKS ---
mock_comfy = types.ModuleType("comfy")
mock_comfy.utils = types.ModuleType("utils")
mock_comfy.utils.ProgressBar = lambda x: type('MockParams', (object,), {'update': lambda self, y: None})()
mock_folder_paths = types.ModuleType("folder_paths")
mock_folder_paths.models_dir = "/tmp" 
mock_folder_paths.add_model_folder_path = lambda x, y: None
mock_folder_paths.get_output_directory = lambda: "debug_output"
sys.modules["comfy"] = mock_comfy
sys.modules["comfy.utils"] = mock_comfy.utils
sys.modules["folder_paths"] = mock_folder_paths

# Ensure we can find nodes.py
sys.path.append(os.getcwd())

try:
    from nodes import Qwen3TTSStageManager
except ImportError as e:
    print(f"Error importing nodes: {e}")
    sys.exit(1)

def test_overlap():
    print(">>> Testing Overlap Behavior <<<")
    
    # 1. Define specific mock model that returns LONG audio for line 9
    class MockModel:
        def __init__(self):
            self.model = type('Inner', (object,), {'tts_model_type': 'base'})()
            
        def generate_voice_design(self, **kwargs):
            text = kwargs.get('text', '')
            print(f"   [MockModel] Generating: '{text[:20]}...'")
            
            # Line 9 content snippet
            if "三七分成" in text:
                print("   !!! Generating EXTRA LONG audio for Line 9 (15s) !!!")
                # 39s - 26s = 13s allowed. We generate 15s.
                duration = 15.0 
            else:
                print("   Generating normal audio (2s)")
                duration = 2.0
                
            sr = 24000
            samples = int(duration * sr)
            audio = np.zeros((samples,), dtype=np.float32)
            # Add a 'signal' (1.0) to detect overlap collision if we want, but logic check is enough
            return [audio], sr

        def generate_voice_clone(self, **kwargs):
            return self.generate_voice_design(**kwargs)

    model = MockModel()

    # 2. Mock Script (Snippet of stage3.srt)
    script_content = """
9 00:00:26,000 --> 00:00:39,000 speaker1: 我告诉告诉你...年终奖三七分成。
10 00:00:39,000 --> 00:00:41,000 speaker2: 怎么才七成啊？
"""
    
    role_def = """
    speaker1 [A]: Mapped A
    speaker2 [B]: Mapped B
    """
    
    sm = Qwen3TTSStageManager()
    dummy_wav = {"waveform": torch.zeros((1, 1, 24000)), "sample_rate": 24000}
    
    if not os.path.exists("debug_output"): os.makedirs("debug_output")
    
    # 3. Run
    # We inspect the internal timeline_events if possible, or just the output log?
    # actually nodes.py prints "Line X ... start=... end=..."? 
    # No, it prints generation finished.
    # The layout logic is internal.
    # But we can verify by checking the SRT output!
    
    modes = ["ignore", "shift_start", "truncate"]
    
    for mode in modes:
        print(f"\n>>> Running StageManager with overlap_handling='{mode}' <<<")
        res = sm.generate_scene(
            model=model,
            script=script_content,
            role_definitions=role_def,
            my_turn_interval=0.0,
            save_to_file=False,
            filename_prefix=f"overlap_test_{mode}",
            role_A_audio=dummy_wav,
            role_B_audio=dummy_wav,
            role_C_audio=dummy_wav,
            role_D_audio=dummy_wav,
            role_E_audio=dummy_wav,
            role_F_audio=dummy_wav,
            role_G_audio=dummy_wav,
            language="Chinese",
            max_new_tokens=512,
            overlap_handling=mode
        )
        
        srt_out = res[-1]
        print(f"--- Generated SRT ({mode}) ---")
        print(srt_out)
        
        # Simple Analysis
        lines = srt_out.strip().split('\n')
        times = [l for l in lines if "-->" in l]
        print(f"Timings: {times}")
        
        if mode == "ignore":
            print("Expectation: Line 2 start (39.000) < Line 1 end")
        elif mode == "shift_start":
            print("Expectation: Line 2 start should be shifted to match Line 1 end (approx 26+15 = 41.000)")
        elif mode == "truncate":
            print("Expectation: Line 1 end should be 39.000 (truncated to fit slot)")
    
    print("\n[Done] Check output above.")

if __name__ == "__main__":
    test_overlap()
