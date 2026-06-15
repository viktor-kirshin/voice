from typing import Literal

from pydantic import BaseModel, Field

from elplatai.clients.ollama_client_chat import get_parsed_data_ollama
from elplatai.clients.vllm_client import get_parsed_data_vllm
from elplatai.config import is_ollama_enabled


class AskLLMAction(BaseModel):
    action: Literal["ask_llm"]
    text: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    quantity: int = Field(default=1, ge=1)
    top_p: float = Field(default=1.0, gt=0, le=1.0)
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0)
    pydantic_schema: str = "get_augment"
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = (
        "none"
    )


def ask_llm(request: AskLLMAction) -> BaseModel | None:
    kwargs = {
        "effort": request.reasoning_effort,
        "quantity": request.quantity,
        "text": request.text,
        "prompt": request.prompt,
        "schema": request.pydantic_schema,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
    }

    parser = get_parsed_data_ollama if is_ollama_enabled() else get_parsed_data_vllm
    return parser(**kwargs)
