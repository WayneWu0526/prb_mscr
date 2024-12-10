from setuptools import setup, find_packages

setup(
    name="prb_mscr",
    version="0.1.0",
    author="WayneWu",
    author_email="zhiweiwu.cn@outlook.com",
    description="A pseudo-rigid-body model library for magnetically controlled soft robots.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/WayneWu0526/prb_mscr.git",  # 项目地址
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",  # 依赖的包
        "scipy>=1.5.0",
        "modern-robotics" 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
