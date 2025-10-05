from setuptools import setup

setup(
    name = "pedestrian_simul",
    version = "2.0",
    author = "Hai H. Vo",
    author_email= "haivo@mit.edu",
    packages=['simulator'],
    install_requires=['numpy', 'matplotlib', 'seaborn', 'jupyter', 'jax', 'jax-md', 'ffmpeg', 'ffmpeg-python', 'equinox']
)
