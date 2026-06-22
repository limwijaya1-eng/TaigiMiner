import os
import json
from typing import List, Dict, Any
import torch
import soundfile as sf

class AudioSegmentationAgent:
    def __init__(self, input_manifest: str = "./taigiminer_data/verified/verified_manifest.json", output_dir: str = "./taigiminer_data/segmented"):
        self.input_manifest = input_manifest
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect local GPU CUDA hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] AI Acceleration Hardware: {str(self.device).upper()}")
        
        # Load official Silero VAD locally via Torch Hub
        print("[Init] Loading local Silero VAD model for Precision Sentence Slicing...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model.to(self.device)
        self.get_speech_timestamps = self.vad_utils[0]
        
        # Segmentation parameters standard in the speech technology industry
        self.min_speech_duration_sec = 2.0  # Ignore if below 2 seconds (too short/short noise)
        self.max_speech_duration_sec = 11.0 # Do not exceed 11 seconds to prevent GPU OOM during ASR/TTS training

    def segment_single_audio(self, video_id: str, audio_path: str) -> List[Dict[str, Any]]:
        """Slices a single large audio file into sentence-level chunks based on VAD."""
        segments_meta = []
        if not os.path.exists(audio_path):
            return []
            
        try:
            # Read .wav audio file
            audio_data, sample_rate = sf.read(audio_path)
            
            # Take the first channel if the audio is stereo
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
                
            # Convert array to PyTorch tensor and move to GPU
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
            
            # Extract real speech timestamps from Silero VAD
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model, 
                sampling_rate=sample_rate
            )
            
            for idx, ts in enumerate(speech_timestamps):
                start_sample = ts['start']
                end_sample = ts['end']
                
                # Calculate segment duration in seconds
                seg_duration = (end_sample - start_sample) / sample_rate
                
                # Strict Duration Filter: Only keep segments within the 2 to 11 seconds range
                if self.min_speech_duration_sec <= seg_duration <= self.max_speech_duration_sec:
                    # Extract audio chunk data from the original array
                    chunk_data = audio_data[start_sample:end_sample]
                    
                    # Determine the storage path for the new segment file
                    seg_filename = f"{video_id}_seg_{idx:04d}.wav"
                    seg_path = os.path.join(self.output_dir, seg_filename)
                    
                    # Save the audio segment to local disk
                    sf.write(seg_path, chunk_data, sample_rate)
                    
                    # Record segment metadata to internal manifest
                    segments_meta.append({
                        "segment_id": f"{video_id}_seg_{idx:04d}",
                        "parent_video_id": video_id,
                        "segment_audio_path": seg_path,
                        "start_time_sec": round(start_sample / sample_rate, 2),
                        "end_time_sec": round(end_sample / sample_rate, 2),
                        "duration_sec": round(seg_duration, 2)
                    })
                    
            return segments_meta
            
        except Exception as e:
            print(f"   [Segment Error] Failed to process file {video_id}: {e}")
            return []

    def run(self) -> List[Dict[str, Any]]:
        print("\n======================================================================")
        print("=== TaigiMiner Audio Segmentation Agent (Precision VAD Slicing) =====")
        print("======================================================================")
        
        if not os.path.exists(self.input_manifest):
            print(f"[Error] Verified manifest file '{self.input_manifest}' not found!")
            return []
            
        with open(self.input_manifest, "r", encoding="utf-8") as f:
            verified_data = json.load(f)
            
        print(f"[Start] Computing segmentation for {len(verified_data)} verified audio files...")
        global_segmented_manifest = []
        
        for idx, video in enumerate(verified_data):
            video_id = video.get("video_id")
            audio_path = video.get("audio_path")
            title = video.get("title", "")
            
            print(f"[{idx+1}/{len(verified_data)}] Slicing: {title[:40]}...")
            
            # Execute VAD-based slicing
            file_segments = self.segment_single_audio(video_id, audio_path)
            
            if file_segments:
                global_segmented_manifest.extend(file_segments)
                print(f"   -> [SUCCESS] Successfully extracted {len(file_segments)} clean sentence segments.")
            else:
                print(f"   -> [INFO] No news sentence segments met the duration qualifications.")
                
        # Save the final audio segment database manifest
        output_manifest_path = os.path.join(self.output_dir, "segmented_manifest.json")
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(global_segmented_manifest, f, ensure_ascii=False, indent=4)
            
        print(f"\n=== [PROCESS SUCCESSFUL - AUDIO SEGMENTATION COMPLETED] ===")
        print(f"Total Verified Input Videos    : {len(verified_data)} files.")
        print(f"Total Extracted Sentence Segments : {len(global_segmented_manifest)} short audio segments.")
        print(f"Database file segmented_manifest saved at: {output_manifest_path}\n")
        return global_segmented_manifest

if __name__ == "__main__":
    agent = AudioSegmentationAgent()
    agent.run()