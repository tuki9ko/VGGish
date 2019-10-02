import pytest

import numpy as np
import librosa
import pumpp
import vggish_keras as vgk


@pytest.mark.parametrize("include_top,compress,expected", [
    (False, False, 512),
    (True, False, 128),
    (True, True, 128)
])
def test_vggish(include_top, compress, expected):
    # setup pump and model
    pump = vgk.get_pump()
    model = vgk.VGGish(pump, include_top=include_top, compress=compress)
    print(pump)
    model.summary()

    # convert data to mel spectrograms and compute embeddings
    sampler = pumpp.SequentialSampler(
        96, *pump.ops, stride=pump.ops[0].n_frames(vgk.params.EXAMPLE_HOP_SECONDS))
    data = sampler(pump(librosa.util.example_audio_file()))
    X = np.concatenate([d[vgk.params.PUMP_INPUT] for d in data], axis=0)
    print(2342423423, X.shape)
    Z = model.predict(X)

    time_points = np.arange(len(X)) * vgk.params.EXAMPLE_HOP_SECONDS

    assert X.shape == (13, 96, 64, 1)
    assert len(time_points) == len(Z)
    assert Z.shape[1:] == (expected,)
    # TODO: test model outputs


@pytest.mark.parametrize("include_top,compress,expected", [
    (False, False, 512),
    (True, False, 128),
    (True, True, 128)
])
def test_get_embedding(include_top, compress, expected):



    ts, Z = vgk.get_embeddings(
        librosa.util.example_audio_file(),
        include_top=include_top, compress=compress)

    assert len(ts) == len(Z)
    assert Z.shape[1:] == (expected,)




    model, pump, sampler = vgk.get_embedding_model(
        include_top=include_top, compress=compress)

    ts, Z = vgk.get_embeddings(
        librosa.util.example_audio_file(),
        model=model, pump=pump, sampler=sampler)

    assert len(ts) == len(Z)
    assert Z.shape[1:] == (expected,)
    # TODO: test model outputs




    func = vgk.get_embedding_function(include_top=include_top, compress=compress)

    ts, Z = func(librosa.util.example_audio_file())

    assert len(ts) == len(Z)
    assert Z.shape[1:] == (expected,)
    # TODO: test model outputs
