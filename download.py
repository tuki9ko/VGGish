import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

DRIVE_URL = 'https://docs.google.com/uc?export=download&id={id}'

FILES = [
    ('vggish_keras/model/audioset_top.h5',
     DRIVE_URL.format(id='1mhqXZ8CANgHyepum7N4yrjiyIg6qaMe6')),

    ('vggish_keras/model/audioset_no_top.h5',
     DRIVE_URL.format(id='16JrWEedwaZFVZYvn1woPKCuWx85Ghzkp')),

    ('vggish_keras/model/audioset_pca_params.npz',
     'https://storage.googleapis.com/audioset/vggish_pca_params.npz'),
]



def download_file(path, url=None):
    if not os.path.isfile(path):
        if url:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            print('Downloading file {} from {} ...'.format(path, url))
            urlretrieve(url, path)
            return None
        print('Could not download {}. No url provided.'.format(path))
    print('File {} already exists.')
    return path


if __name__ == '__main__':
    files = [download_file(f, url) for f, url in FILES]
