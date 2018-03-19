from setuptools import setup
import sys

# Only support Python 3
if sys.version_info.major != 3:
    print(f'WARNING: This package only officially support Python 3, the current version is Python {sys.version_info.major}. The installation will likely fail. ')

setup(name='gym_maze',
      install_requires=['gym', 
                        'numpy', 
                        'matplotlib', 
                        'scikit-image', 
                        'jupyterlab'],
      description='gym-maze: A customizable gym environment for maze and gridworld',
      author='Xingdong Zuo',
      url='https://github.com/zuoxingdong/gym-maze',
      version='0.1'
)
