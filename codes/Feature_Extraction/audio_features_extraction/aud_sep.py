import os
import subprocess

VLC_PATH = r"C:\Program Files\VideoLAN\VLC\vlc.exe"         # Path of VLC executable

VIDEO_FOLDER = r"C:\Users\sambi\Documents\AP1"
AUDIO_FOLDER = r"C:\Users\sambi\Documents\AP1_audio"

os.makedirs(AUDIO_FOLDER, exist_ok=True)

video_exts = (".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv")       # Supported video formats

for file in os.listdir(VIDEO_FOLDER):
    if file.lower().endswith(video_exts):
        video_path = os.path.join(VIDEO_FOLDER, file)
        audio_name = os.path.splitext(file)[0] + ".wav"
        audio_path = os.path.join(AUDIO_FOLDER, audio_name)

        cmd = [
            VLC_PATH,
            "-I", "dummy",
            video_path,
            "--no-video",
            "--audio-filter=converter",
            "--sout=#transcode{acodec=s16l,channels=1,samplerate=16000}:std{access=file,mux=wav,dst=" + audio_path + "}",
            "vlc://quit"
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Audio extraction completed.")
