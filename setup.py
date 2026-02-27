from setuptools import setup, find_packages
import os

# 读取 README.md 作为长描述
with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SigflowVelo",
    version="0.1.1",  # 记得每次上传前更新版本号
    description="A Deep Learning framework for Cell-Cell Communication Velocity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scanpy==1.9.8",
        "scvelo==0.3.2",
        "torch==2.1.2",
        "matplotlib==3.7.5",
        "scipy==1.10.1"
    ],
    author="Your Name",
    author_email="your.email@example.com", 
    url="https://github.com/yourusername/SigflowVelo", 
    python_requires='>=3.8',
)