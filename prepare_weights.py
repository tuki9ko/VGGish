import os

def load_ckpt(file):
    # https://github.com/tensorflow/tensorflow/blob/892e94736dbda5518e59e66b4a0e8b85f58e014b/tensorflow/python/tools/inspect_checkpoint.py#L57
    from tensorflow.python import pywrap_tensorflow
    reader = pywrap_tensorflow.NewCheckpointReader(file)

    return {
        key: reader.get_tensor(key)
        for key in reader.get_variable_to_shape_map()
    }

def convert_vggish(ckpt_file, pca_file, h5_file):
    data = load_ckpt(ckpt_file)

    for k in list(data):
        data[(k.replace('vggish/', '')
               .replace('weights', 'kernel:0')
               .replace('biases', 'bias:0'))] = data.pop(k)

    import numpy as np
    with np.load(pca_file) as d:
        data['postprocess/pca_matrix:0'] = d['pca_eigen_vectors']
        data['postprocess/pca_means:0'] = d['pca_means']

    for k in list(data):
        p, n = k.rsplit('/', 1)
        data[os.path.join(p, p, n)] = data.pop(k)

    for k in sorted(data):
        print(k, data[k].shape)

    print()

    import h5py
    f = h5py.File(h5_file)

    f.update(data)
    f.visititems(lambda k, v: print(k, k in data) if isinstance(v, h5py.Dataset) else None)

    raise NotImplementedError('Still need to copy weight info. Its probs easier to just load the model....')


if __name__ == '__main__':
    # 'vggish_audioset_weights (1).h5'
    convert_vggish('vggish_model.ckpt', 'vggish_pca_params.npz', 'vggish_weights.h5')
