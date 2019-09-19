import pytest

import librosa
import vggish_keras as vgk

def test_vggish():
    # setup pump and model
    INPUT = 'mel/mag'
    pump = vgk.get_pump()
    model = vgk.VGGish(pump[INPUT])
    
    print(pump)
    model.summary()

    # convert data to mel spectrograms and compute embeddings
    data = pump.transform(librosa.util.example_audio_file())
    X = data[INPUT]
    emb = model.predict(X)

    assert X.shape == (1, 6145, 64, 1)
    assert emb.shape[1] == 512
