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

repo = "TheBloke/samantha-mistral-7B-GGUF"
model = "samantha-mistral-7b.Q4_0.gguf"

path = hf_hub_download(repo_id=repo, filename=model)
llama = LlamaCpp(model_path=path, n_gpu_layers=40, n_batch=512, streaming=True)


add_routes(app, Llama2Chat(llm=llama), path="/chat")


def main():
    ngrok_tunnel = ngrok.connect(8000)
    logger.info('Public URL: %s', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)


if __name__ == '__main__':

    main()
