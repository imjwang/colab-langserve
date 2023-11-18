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
