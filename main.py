import librosa

from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

audio_input = 'cut_listz.mp3'
name = audio_input.split(".")[0]
midi_intermediate_filename = f'{name}.mid'
transcriptor = PianoTranscription(device='cpu', checkpoint_path='./model.pth')
audio, _ = librosa.core.load(str(audio_input), sr=sample_rate)
transcriptor.transcribe(audio, midi_intermediate_filename)