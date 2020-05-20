# VGGish: A VGG-like audio classification model

This repository provides a VGGish model, implemented in Keras with tensorflow backend (since `tf.slim` is [deprecated](https://github.com/tensorflow/tensorflow/issues/16182#issuecomment-372397483), I think we should have an up-to-date interface). This repository is developed
based on the model for [AudioSet](https://research.google.com/audioset/index.html).
For more details, please visit the [slim version](https://github.com/tensorflow/models/tree/master/research/audioset).



## Install

```bash
pip install vggish-keras
```
Weights will be downloaded the first time they are requested. You can also run `python -m vggish_keras.download_helpers.download_weights` which will download them.

## Usage
Basic - simple & efficient method:
```python
import librosa
import numpy as np
import vggish_keras as vgk

# loads the model once and provides a simple function that takes in `filename` or `y, sr`
compute = vgk.get_embedding_function(hop_duration=0.25)
# model, pump, and sampler are available as attributes
compute.model.summary() # take a peak at the model

# compute from filename
Z = compute(librosa.util.example_audio_file())

# compute from pcm
y, sr = librosa.load(librosa.util.example_audio_file())
Z, ts = compute(y=y, sr=sr)
```

Alternatives - using the under-the-hood helper functions:
```python
# get the embeddings - WARNING: it instantiates a new model each time
Z, ts = vgk.get_embeddings(librosa.util.example_audio_file(), hop_duration=0.25)

# create model, pump, sampler once and pass to vgk.get_embeddings
# - more typing :'(
model, pump, sampler = vgk.get_embedding_model(hop_duration=0.25)
Z, ts = vgk.get_embeddings(
    librosa.util.example_audio_file(),
    model=model, pump=pump, sampler=sampler)
```

Manually, using the keras model and pump directly:
```python
import librosa
import numpy as np
import vggish_keras as vgk

# define the model
pump = vgk.get_pump()
model = vgk.VGGish(pump)
sampler = vgk.get_sampler(pump)

# transform audio into VGGish embeddings
filename = librosa.util.example_audio_file()
X = np.concatenate([
    x[vgk.params.PUMP_INPUT]
    for x in sampler(pump(filename))])
Z = model.predict(X)

# calculate timestamps
ts = vgk.get_timesteps(Z, pump, sampler)
assert Z.shape == (13, 512)
```

## Reference:

* Gemmeke, J. et. al.,
  [AudioSet: An ontology and human-labelled dataset for audio events](https://research.google.com/pubs/pub45857.html),
  ICASSP 2017

* Hershey, S. et. al.,
  [CNN Architectures for Large-Scale Audio Classification](https://research.google.com/pubs/pub45611.html),
  ICASSP 2017

I include a weight conversion script in [download_helpers/convert_ckpt.py](https://github.com/beasteers/VGGish/blob/master/vggish_keras/download_helpers/convert_ckpt.py) which shows how I converted the weights from `.ckpt` to `.h5` for [those](https://github.com/DTaoo/VGGish/issues/6) that are interested.

## TODO
 - currently, parameters (sample rate, hop size, etc) can be changed globally via `vgk.params` - I'd like to allow for parameter overrides to be passed to `vgk.VGGish`
 - currently it relies on https://github.com/bmcfee/pumpp/pull/123. Once merged, remove custom github install location
