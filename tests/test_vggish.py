import pytest

import numpy as np
import librosa
import pumpp
import vggish_keras as vgk

TEST_PARAMS = [
    (False, False, 512),
    (True, False, 128),
    (True, True, 128)
]

def check_outputs(Z, ts, expected_zdim, X=None, n_frames=13, expected_xdim=(96, 64, 1)):
    if X is not None:
        assert X.shape == (n_frames,) + expected_xdim
        assert len(X) == len(Z)
    assert Z.shape == (n_frames, expected_zdim)
    assert len(ts) == len(Z)
    assert np.allclose(ts, np.arange(len(Z)) * vgk.p.EXAMPLE_HOP_SECONDS)
    # TODO: test model outputs

@pytest.mark.parametrize("include_top,compress,expected", TEST_PARAMS)
def test_vggish(include_top, compress, expected):
    # setup pump and model
    pump = vgk.get_pump()
    model = vgk.VGGish(pump, include_top=include_top, compress=compress)
    sampler = vgk.get_sampler(pump)
    print(pump)
    # model.summary()

    # convert data to mel spectrograms and compute embeddings
    data = sampler(pump(librosa.util.example_audio_file()))
    X = np.concatenate([d[vgk.params.PUMP_INPUT] for d in data], axis=0)
    Z = model.predict(X)

    ts = vgk.get_timesteps(Z, pump)
    check_outputs(Z, ts, expected, X)

@pytest.mark.parametrize("include_top,compress,expected", TEST_PARAMS)
def test_get_embedding(include_top, compress, expected):
    Z, ts = vgk.get_embeddings(
        librosa.util.example_audio_file(),
        include_top=include_top, compress=compress)

    check_outputs(Z, ts, expected)

@pytest.mark.parametrize("include_top,compress,expected", TEST_PARAMS)
def test_get_embedding_model(include_top, compress, expected):
    model, pump, sampler = vgk.get_embedding_model(
        include_top=include_top, compress=compress)

    Z, ts = vgk.get_embeddings(
        librosa.util.example_audio_file(),
        model=model, pump=pump, sampler=sampler)

    check_outputs(Z, ts, expected)

@pytest.mark.parametrize("include_top,compress,expected", TEST_PARAMS)
def test_get_embedding_function(include_top, compress, expected):
    compute = vgk.get_embedding_function(include_top=include_top, compress=compress)
    Z, ts = compute(librosa.util.example_audio_file())

    check_outputs(Z, ts, expected)
