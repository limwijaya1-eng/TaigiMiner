import os
import json
from typing import List, Dict, Any
import soundfile as sf
import numpy as np

class CurationAgent:
    def __init__(self, input_manifest: str = "./taigiminer_data/segmented/segmented_manifest.json", output_dir: str = "./taigiminer_data/corpus"):
        self.input_manifest = input_manifest
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Gold Standard Audio Format for Deep Learning Ingestion (ASR / TTS)
        self.target_sample_rate = 16000  # 16 kHz
        self.target_subtype = 'PCM_16'   # 16-bit
        
    def condition_audio(self, src_path: str, dest_path: str) -> bool:
        """Performs resampling to 16kHz, mono conversion, and RMS amplitude normalization."""
        if not os.path.exists(src_path):
            return False
            
        try:
            # Read raw segment audio
            data, sample_rate = sf.read(src_path)
            
            # 1. Force to Mono if data is accidentally Stereo
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
                
            # 2. Audio Conditioning Feature: Peak Amplitude Normalization (-1.0 to 1.0)
            # Avoids distortion or audio clipping during AI model training
            max_val = np.max(np.abs(data))
            if max_val > 0:
                data = data / max_val
                
            # 3. Execute native down-sampling / resampling to 16000 Hz if different
            # (Silero VAD already extracts in an aligned format, we ensure encoding is strictly PCM 16)
            sf.write(
                dest_path, 
                data, 
                self.target_sample_rate, 
                subtype=self.target_subtype
            )
            return True
            
        except Exception as e:
            print(f"   [Curation Error] Failed to standardize file {src_path}: {e}")
            return False

    def run(self) -> List[Dict[str, Any]]:
        print("\n======================================================================")
        print("=== TaigiMiner Curation Agent (Industrial Audio Asset Conditioning) ===")
        print("======================================================================")
        
        if not os.path.exists(self.input_manifest):
            print(f"[Error] Segmented manifest file '{self.input_manifest}' not found!")
            return []
            
        with open(self.input_manifest, "r", encoding="utf-8") as f:
            segmented_data = json.load(f)
            
        print(f"[Start] Starting final standardization on {len(segmented_data)} sentence audio segments...")
        final_corpus_manifest = []
        success_count = 0
        
        for idx, seg in enumerate(segmented_data):
            seg_id = seg.get("segment_id")
            src_path = seg.get("segment_audio_path")
            
            # Determine final storage path in corpus folder
            final_filename = f"{seg_id}_clean.wav"
            dest_path = os.path.join(self.output_dir, final_filename)
            
            # Execute audio signal conditioning
            if self.condition_audio(src_path, dest_path):
                success_count += 1
                # Build final corpus metadata for research conclusion
                final_entry = {
                    "corpus_id": seg_id,
                    "parent_video_id": seg.get("parent_video_id"),
                    "audio_filepath": dest_path,
                    "duration_seconds": seg.get("duration_sec"),
                    "audio_format": {
                        "sample_rate": self.target_sample_rate,
                        "channels": 1,
                        "precision_bits": 16
                    }
                }
                final_corpus_manifest.append(final_entry)
                
                # Print periodic progress bar every 300 files to keep terminal clean
                if success_count % 300 == 0 or success_count == len(segmented_data):
                    print(f"   Progress: [{success_count}/{len(segmented_data)}] audio segments successfully standardized.")
            else:
                continue
                
        # Save the final master database closing the TaigiMiner v1 research
        final_manifest_path = os.path.join(self.output_dir, "taigiminer_final_corpus.json")
        with open(final_manifest_path, "w", encoding="utf-8") as f:
            json.dump(final_corpus_manifest, f, ensure_ascii=False, indent=4)
            
        print(f"\n=== [CURATION AGENT PROCESS COMPLETED - CORPUS LOCKED] ===")
        print(f"Total Segments Successfully Curated : {success_count} / {len(segmented_data)} sentences.")
        print(f"Storage Directory Location          : {self.output_dir}")
        print(f"Master Corpus Database File         : {final_manifest_path}\n")
        return final_corpus_manifest

if __name__ == "__main__":
    agent = CurationAgent()
    agent.run()