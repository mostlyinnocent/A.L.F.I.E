from pathlib import Path

def delete_audio(file_path):
  audio_file = Path(file_path)

  audio_file.unlink()
  print(f"deleted: {audio_file}")