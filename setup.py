''''''
import os
import sys

import setuptools

import download

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # exclude the weight files in sdist
    weight_files = []
else:
    weight_files = [download.download_file(f, url) for f, url in download.FILES]
    weight_files = [f for f in weight_files if f and os.path.splitext(f)[1] == '.h5']


setuptools.setup(name='vggish_keras',
                 version='0.0.1',
                 description='VGGish in Keras.',
                 long_description=open('README.md').read().strip(),
                 author='Bea Steers',
                 author_email='bea.steers@gmail.com',
                 # url='http://path-to-my-packagename',
                 packages=setuptools.find_packages(),
                 # py_modules=['packagename'],
                 package_data={'vggish': weight_files},
                 # include_package_data=True,
                 install_requires=[],
                 license='MIT License',
                 zip_safe=False,
                 keywords='vggish audio audioset keras tensorflow')
