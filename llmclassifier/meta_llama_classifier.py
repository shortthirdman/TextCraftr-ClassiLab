import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

load_dotenv()


class ClassificationOutput(BaseModel):
    category: Literal["news", "clickbait"]


llm_client = ChatOpenAI(
    openai_api_base=os.environ.get("LLM_BASE_URL"),
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    openai_api_key=os.environ.get("LLM_API_KEY"),
    temperature=0,
    max_retries=2,
)

constrained_llm = llm_client.with_structured_output(ClassificationOutput)

messages = [
    SystemMessage(
        content="Classify the following text into one of the predefined categories: news or clickbait"
    ),
    HumanMessage(content="You won't believe what happened next!"),
]
prediction = constrained_llm.invoke(messages)

print(prediction)