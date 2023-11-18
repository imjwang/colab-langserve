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

repo = "TheBloke/Llama-2-13B-chat-GGUF"
model = "llama-2-13b-chat.Q6_K.gguf"

path = hf_hub_download(repo_id=repo, filename=model)
llama = LlamaCpp(model_path=path, n_gpu_layers=43, n_batch=512, stream=False)


add_routes(app, Llama2Chat(llm=llama), path="/chat")

add_routes(app, llama, path="/plain")


def main():
    ngrok_tunnel = ngrok.connect(8000)
    logger.info('Public URL: %s', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)


if __name__ == '__main__':

    main()
