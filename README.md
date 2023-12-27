# colab-langserve

Make sure GPU is selected.

## Setup

```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install -r requirements.txt
```

This step is needed for now. Manually install langchain experimental:

```shell
git clone https://github.com/langchain-ai/langchain.git
cd langchain/libs/experimental/
pip install -e .
```

```shell
python main.py
```

Save the url from logger

### Sources

colab ngrok hosting https://ko-fi.com/yeyuh (https://colab.research.google.com/drive/16S5dJIiYSprRQQCSccW7o1Slqw-NOifl?usp=sharing\#scrollTo=S1t4hv0_AZL_)
Do public READMEs need citing?
