import librosa
import numpy as np
import cv2

def generate_spectrogram(audio_array: np.ndarray, sampling_rate: int) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=audio_array, sr=sampling_rate,
                                        n_mels=128, fmax=8000)
    S_DB = librosa.power_to_db(S, ref=np.max)

    S_DB -= S_DB.min()
    S_DB /= S_DB.max()
    
    S_resized = cv2.resize(S_DB, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    return S_resized