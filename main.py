import librosa
import os
import pathlib

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio
from synthviz import create_video

audio_input = 'my_audio.mp3'
midi_intermediate_filename = 'transcription.mid'
video_filename = 'output.mp4'

transcriptor = PianoTranscription(device='cuda', checkpoint_path='./model.pth')
audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
transcribed_dict = transcriptor.transcribe(audio, midi_intermediate_filename)
create_video(input_midi=midi_intermediate_filename, video_filename=video_filename)
