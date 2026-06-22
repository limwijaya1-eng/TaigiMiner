import os
import json
import re
from typing import List, Dict, Any
import torch
import soundfile as sf

class LanguageVerificationAgent:
    def __init__(self, input_manifest: str = "./taigiminer_data/raw/discovery_manifest.json", output_dir: str = "./taigiminer_data/verified"):
        self.input_manifest = input_manifest
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize local GPU CUDA hardware acceleration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] AI Acceleration Hardware: {str(self.device).upper()}")
        
        # Load official Silero VAD locally via Torch Hub (Lightweight, Accurate, & FOSS)
        print("[Init] Loading local Silero VAD model from Torch Hub...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model.to(self.device)
        self.get_speech_timestamps = self.vad_utils[0]
        
        # Source Trust Matrix designed for your research framework
        self.TRUSTED_SOURCES = {
            "公視台語新聞": 1.0,
            "民視台語新聞": 1.0,
            "華視台語新聞": 1.0,
            "三立新聞": 1.0,
            "台語新聞": 0.8,
            "台語新聞完整版": 0.8,
            "台語新聞直播": 0.8
        }
        
        # Data Cleaning Statistical Report Tracker for answering RQ1 in the Paper
        self.report_stats = {
            "total_raw_mined_input": 0,
            "dropped_by_music_metadata_rule": 0,
            "dropped_by_hard_speech_ratio_limit": 0,
            "dropped_by_low_final_vs": 0,
            "final_verified_keep": 0
        }

    def evaluate_music_filter(self, title: str) -> float:
        """Rule 1: Checks if the video is a song based on title tokens."""
        song_flags = ["MV", "OFFICIAL MV", "MUSIC VIDEO", "歌曲", "音樂", "歌詞", "KTV", "卡拉OK", "演唱會", "情歌"]
        for flag in song_flags:
            if flag in title.upper():
                return 0.0  # REJECT if a song is detected
        return 1.0  # PASS

    def evaluate_source_trust(self, title: str, source_keyword: str) -> float:
        """Rule 3: Calculates SourceTrust score based on official broadcasting station validation."""
        combined_text = (title + " " + source_keyword).upper()
        
        # PRIMARY ANCHOR: If it contains a combination of official Taiwan TV station names
        # Matching the whitelist: 公視, 民視, 華視, 三立, PTS, FTV, CTS, SETN
        official_channels = ["公視", "民視", "華視", "三立", "PTS", "FTV", "CTS", "SETN"]
        for channel in official_channels:
            if channel in combined_text:
                return 1.0  # Pure Official News (High Reliability)
                
        # SECONDARY CATEGORY: If it contains generic news keywords
        generic_news = ["台語新聞", "閩南語新聞", "新聞直播", "新聞完整版"]
        for news_key in generic_news:
            if news_key in combined_text:
                return 0.8  # Weighted score for general news keywords
                
        return 0.5  # Fallback for general / unverified channels

    def calculate_real_speech_ratio(self, audio_path: str) -> float:
        """Rule 2: Calculates Real Speech Duration Ratio using Silero VAD."""
        try:
            # Read local .wav audio file
            audio_data, sample_rate = sf.read(audio_path)
            
            # Silero VAD requires a sample rate of 16000Hz or 8000Hz
            # If audio_data has stereo channels (2D), take the first channel only
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
                
            total_duration = len(audio_data) / sample_rate
            if total_duration == 0:
                return 0.0
                
            # Convert array to PyTorch tensor and move to GPU/CPU
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
            
            # Extract timestamps of human speech segments
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model, 
                sampling_rate=sample_rate
            )
            
            # Calculate accumulated real speech duration (in seconds)
            total_speech_samples = sum((ts['end'] - ts['start']) for ts in speech_timestamps)
            total_speech_duration = total_speech_samples / sample_rate
            
            speech_ratio = total_speech_duration / total_duration
            return min(1.0, max(0.0, speech_ratio))
            
        except Exception as e:
            print(f"   [VAD Error] Failed to analyze acoustic features of file {audio_path}: {e}")
            return 0.0

    def process_verification(self) -> List[Dict[str, Any]]:
        print("\n======================================================================")
        print("=== TaigiMiner Verification Agent v6 (Silero VAD & Source Trust) ===")
        print("======================================================================")
        
        if not os.path.exists(self.input_manifest):
            print(f"[Error] Index file '{self.input_manifest}' not found!")
            return []
            
        with open(self.input_manifest, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
            
        self.report_stats["total_raw_mined_input"] = len(manifest_data)
        verified_dataset = []
        detailed_logs = []
        
        for idx, video in enumerate(manifest_data):
            title = video.get("title", "")
            audio_path = video.get("audio_path", "")
            source_keyword = video.get("source_keyword", "")
            
            print(f"\n[{idx+1}/{len(manifest_data)}] Evaluating: {title[:50]}...")
            
            # STAGE 1: METADATA MUSIC FILTER
            m_score = self.evaluate_music_filter(title)
            if m_score == 0.0:
                print("   -> [DECISION: REJECT] Detected as music/song via title metadata.")
                self.report_stats["dropped_by_music_metadata_rule"] += 1
                detailed_logs.append({"video_id": video.get("video_id"), "title": title, "decision": "REJECT_MUSIC"})
                continue
                
            if not os.path.exists(audio_path):
                print(f"   -> [SKIP] Physical audio file not found on local disk.")
                continue
                
            # STAGE 2: REAL SPEECH RATIO VIA SILERO VAD
            speech_ratio = self.calculate_real_speech_ratio(audio_path)
            
            # Evaluate Hard Cut-off Limit (< 0.60 REJECT)
            if speech_ratio < 0.60:
                print(f"   -> [DECISION: REJECT] Hard Limit: Speech Ratio too low ({speech_ratio:.2f} < 0.60).")
                self.report_stats["dropped_by_hard_speech_ratio_limit"] += 1
                detailed_logs.append({"video_id": video.get("video_id"), "title": title, "decision": "REJECT_HARD_VAD_LIMIT", "speech_ratio": speech_ratio})
                continue
                
            # STAGE 3: SOURCE TRUST WEIGHTING
            source_trust = self.evaluate_source_trust(title, source_keyword)
            
            # YOUR ROBUST ACADEMIC FINAL FORMULA
            vs_score = (0.7 * speech_ratio) + (0.3 * source_trust)
            
            print(f"   -> Metrics Analysis: SpeechRatio={speech_ratio:.2f}, SourceTrust={source_trust:.1f} | FINAL VS: {vs_score:.2f}")
            
            # Evaluate Final Passing Threshold (VS >= 0.75 KEEP)
            if vs_score >= 0.75:
                video["verification_score"] = vs_score
                video["speech_ratio"] = speech_ratio
                video["source_trust"] = source_trust
                verified_dataset.append(video)
                self.report_stats["final_verified_keep"] += 1
                detailed_logs.append({"video_id": video.get("video_id"), "title": title, "decision": "KEEP", "score": vs_score})
                print("   -> [STATUS: KEEP] Validated and secured for downstream corpus slicing.")
            else:
                print(f"   -> [DECISION: REJECT] VS Score ({vs_score:.2f}) below eligibility threshold of 0.75.")
                self.report_stats["dropped_by_low_final_vs"] += 1
                detailed_logs.append({"video_id": video.get("video_id"), "title": title, "decision": "REJECT_LOW_VS", "score": vs_score})
                
        # Save the clean database index of the verified corpus
        output_manifest_path = os.path.join(self.output_dir, "verified_manifest.json")
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(verified_dataset, f, ensure_ascii=False, indent=4)
            
        # Save empirical quantitative report for the Results Chapter Table in the Paper (Answering RQ1)
        report_path = os.path.join(self.output_dir, "rq1_verification_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"summary_statistics": self.report_stats, "logs": detailed_logs}, f, ensure_ascii=False, indent=4)
            
        print(f"\n=== [PROCESS COMPLETED - FINAL SECURED VERSION] ===")
        print(json.dumps(self.report_stats, indent=4))
        print(f"Database file verified_manifest saved at: {output_manifest_path}\n")
        return verified_dataset

if __name__ == "__main__":
    agent = LanguageVerificationAgent()
    agent.process_verification()