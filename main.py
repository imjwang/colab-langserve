from huggingface_hub import hf_hub_download
from fastapi import FastAPI
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from pydantic import BaseModel
from langchain.llms import LlamaCpp
import logging
from langserve import add_routes
from langchain_experimental.chat_models import Llama2Chat
from langchain.llms import LlamaCpp
from langchain.chat_models import ChatOpenAI


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

repo = "TheBloke/Llama-2-7B-Chat-GGUF"
model = "llama-2-7b-chat.Q2_K.gguf"

path = hf_hub_download(repo_id=repo, filename=model)
# llama = LlamaCpp(model_path=path, n_gpu_layers=43, n_batch=512, stream=False)


add_routes(app, Llama2Chat(llm=LlamaCpp(model_path=path,
           n_gpu_layers=43, n_batch=512, stream=False)), path="/chat_llama_temp")

add_routes(app, LlamaCpp(model_path=path, n_gpu_layers=43,
           n_batch=512, stream=False), path="/plain")

add_routes(app, ChatOpenAI(), path="/chat")

add_routes


def main():
    ngrok_tunnel = ngrok.connect(8000)
    logger.info('Public URL: %s', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)


if __name__ == '__main__':
    main()
