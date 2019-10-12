# VGGish: A VGG-like audio classification model 

This repository provides a VGGish model, implemented in Keras with tensorflow backend (since `tf.slim` is [deprecated](https://github.com/tensorflow/tensorflow/issues/16182#issuecomment-372397483), I think we should have an up-to-date interface). This repository is developed 
based on the model for [AudioSet](https://research.google.com/audioset/index.html). 
For more details, please visit the [slim version](https://github.com/tensorflow/models/tree/master/research/audioset).



## Install

```bash
pip install vggish-keras
```
Weights will be downloaded the first time they are requested. You can also run `python -m vggish_keras.download_helpers.download_weights` which will download them.

NOTE: Currently - this relies on a pending change to `pumpp` in https://github.com/bmcfee/pumpp/pull/123. To get those changes, you need 

```bash
pip install git+https://github.com/beasteers/pumpp@tf_keras
```

## Usage
```python
import librosa
import numpy as np
import vggish_keras as vgk

# define the model
pump = vgk.get_pump()
model = vgk.VGGish(pump)

# transform audio into VGGish embeddings without fc layers
X = pump.transform(librosa.util.example_audio_file())[vgk.params.PUMP_INPUT]
X = np.concatenate([X]*5)
Z = model.predict(X)

# calculate timestamps
op = pump['mel']
ts = np.arange(len(Z)) / op.sr * op.hop_length
assert Z.shape == (5, 512)
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
