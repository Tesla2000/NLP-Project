import pandas as pd
import numpy as np
df = pd.read_parquet("test_spectrograms.parquet")

# Retrieve the first spectrogram and convert it back to a NumPy array
spectrogram_list = df.loc[0, 'spectrogram']
spectrogram = np.array(spectrogram_list)

# Verify the shape of the spectrogram
print(spectrogram.shape)