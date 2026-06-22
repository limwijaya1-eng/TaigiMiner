import os
import json
from typing import List, Dict, Any
import yt_dlp

class DiscoveryAgent:
    def __init__(self, output_dir: str = "./taigiminer_data/raw"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # Global set to prevent the same video_id from being downloaded multiple times in this session
        self.downloaded_ids = set()
        
    def load_keywords_from_file(self, file_path: str = "keywords.txt") -> List[str]:
        """Reads a collection of conversational keywords from an external file."""
        if not os.path.exists(file_path):
            print(f"[Error] File '{file_path}' not found!")
            print("Please create a 'keywords.txt' file first in the same folder.")
            return []
        
        with open(file_path, "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f.readlines() if line.strip()]
        print(f"[Discovery] Successfully loaded {len(keywords)} conversational keywords from {file_path}")
        return keywords

    def harvest_all_conversational_videos(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Harvests all available conversational videos from the keyword list from scratch."""
        discovered_manifest = []
        yt_dlp_path = r"C:\Users\fcu\AppData\Roaming\Python\Python312\Scripts\yt-dlp.exe"
        outtmpl = os.path.join(self.output_dir, "%(id)s.%(ext)s")
        
        # Fetch top 40 candidates per keyword to maximize thorough harvesting results
        max_results_per_query = 40
        
        # Native Python API configuration for yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'ignoreerrors': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }]
        }
        
        print(f"\n=== Starting Conversational Audio Harvesting Process (Clean & Quota-Free) ===")
        
        for keyword in keywords:
            print(f"\n[Search] Hunting thoroughly with keyword: '{keyword}'...")
            search_query = f"ytsearch{max_results_per_query}:{keyword}"
            
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Fetch metadata in memory without downloading the file immediately
                    info_dict = ydl.extract_info(search_query, download=False)
                    
                    if not info_dict or 'entries' not in info_dict:
                        print(f"   -> Search for '{keyword}' returned no results.")
                        continue
                        
                    for entry in info_dict['entries']:
                        if not entry:
                            continue
                            
                        video_id = entry.get('id')
                        title = entry.get('title')
                        duration = entry.get('duration')
                        
                        # 1. Internal session de-duplication check to prevent duplicate files
                        if video_id in self.downloaded_ids:
                            continue
                            
                        # 2. Duration Filter: Limit from 1 Minute (60s) to 20 Minutes (1200s)
                        if duration and (60 <= duration <= 1200):
                            audio_path = os.path.join(self.output_dir, f"{video_id}.wav")
                            
                            print(f"   [Match Found] {title} [{video_id}] ({duration}s). Downloading...")
                            
                            # Execute actual download specific to this video URL
                            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                            
                            # Physical validation whether FFmpeg successfully created the .wav file on local disk
                            if os.path.exists(audio_path):
                                info = {
                                    "video_id": video_id,
                                    "title": title,
                                    "duration": duration,
                                    "source_keyword": keyword,
                                    "audio_path": audio_path,
                                    "subtitle_path": os.path.join(self.output_dir, f"{video_id}.zh-Tw.vtt")
                                }
                                discovered_manifest.append(info)
                                self.downloaded_ids.add(video_id) # Lock this ID to prevent duplicates
                                print(f"   [Success] Collected #{len(discovered_manifest)}: {audio_path}")
                            else:
                                print(f"   [Warning] Failed to convert audio for ID [{video_id}]. FFmpeg returned an error.")
                                
            except Exception as e:
                print(f"   [Error] Technical issue on query '{keyword}': {e}")
                
        return discovered_manifest

    def run(self) -> List[Dict[str, Any]]:
        """Main execution function for Discovery Agent v3 (Purely from Scratch)."""
        print("======================================================================")
        print("=== TaigiMiner Discovery Agent v3 (Pure Clean & Limitless) ======")
        print("======================================================================")
        
        # Ignore and overwrite old manifest files to ensure data is completely clean from the start
        manifest_path = os.path.join(self.output_dir, "discovery_manifest.json")

        # 1. Load keywords from keywords.txt
        keywords = self.load_keywords_from_file("keywords.txt")
        if not keywords:
            return []
            
        # 2. Run comprehensive data harvesting from scratch
        manifest_data = self.harvest_all_conversational_videos(keywords)
        
        # 3. Save fresh output to JSON manifest file
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=4)
            
        print(f"\n=== [PROCESS SUCCESSFUL] Clean Exploration of Raw Corpus Completed ===")
        print(f"Total videos successfully harvested and landed on local disk: {len(manifest_data)} videos.")
        print(f"New manifest database file saved at: {manifest_path}\n")
        return manifest_data

if __name__ == "__main__":
    agent = DiscoveryAgent(output_dir="./taigiminer_data/raw")
    manifest = agent.run()