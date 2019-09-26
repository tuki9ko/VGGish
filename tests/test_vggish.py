import pytest

import librosa
import vggish_keras as vgk


@pytest.mark.parametrize("include_top,compress,expected", [
    (False, False, (1, 512)),
    (True, False, (1, 128)),
    (True, True, (1, 128))
])
def test_vggish(include_top, compress, expected):
    # setup pump and model
    pump = vgk.get_pump()
    model = vgk.VGGish(pump)

    print(pump)
    model.summary()

    # convert data to mel spectrograms and compute embeddings
    data = pump.transform(librosa.util.example_audio_file())
    X = data[vgk.params.PUMP_INPUT]
    emb = model.predict(X)

    assert X.shape == (1, 6145, 64, 1)
    assert emb.shape == (1, 512)
