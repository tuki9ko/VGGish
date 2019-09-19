import pytest

import librosa
import vggish_keras as vgk

def test_vggish():
    # setup pump and model
    pump = vgk.get_pump()
    model = vgk.VGGish(pump[vgk.params.PUMP_INPUT])

    print(pump)
    model.summary()

    # convert data to mel spectrograms and compute embeddings
    data = pump.transform(librosa.util.example_audio_file())
    X = data[vgk.params.PUMP_INPUT]
    emb = model.predict(X)

    assert X.shape == (1, 6145, 64, 1)
    assert emb.shape[1] == 512
