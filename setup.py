from setuptools import setup, find_packages

setup(
    name="glitchminer",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "torch",
        # 添加其他依赖项
    ],
    author="Zihui Wu, adapted by Scrattlebeard",
    description="A tool for mining glitch tokens in gpt-oss-20b",
    url="https://github.com/scrattlebeard/glitchminer",
)
