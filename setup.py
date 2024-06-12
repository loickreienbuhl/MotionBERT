from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="MotionBERT",
    version="0.1.0",
    author="forkedByLoic",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'tensorboardX',
        'tqdm',
        'easydict',
        'prettytable',
        'chumpy',
        'opencv-python',
        'imageio-ffmpeg',
        'matplotlib==3.1.1',
        'roma',
        'ipdb',
        'pytorch-metric-learning',
        'smplx[all]'
    ],
    python_requires='>=3.8, <3.11',
    include_package_data=True,
)
