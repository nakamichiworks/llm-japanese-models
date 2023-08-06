from typing import Optional, Iterator, Callable, Set

import llm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@llm.hookimpl
def register_models(register: Callable[[llm.Model, Optional[Set[str]]], None]):
    register(OpenCalm("open-calm-small"), aliases=("open-calm-160m",))
    register(OpenCalm("open-calm-medium"), aliases=("open-calm-400m",))
    register(OpenCalm("open-calm-large"), aliases=("open-calm-830m",))
    register(OpenCalm("open-calm-1b"))
    register(OpenCalm("open-calm-3b"))
    register(OpenCalm("open-calm-7b"))


class OpenCalm(llm.Model):
    def __init__(self, model_id: str):
        self.model_id = model_id

    def __str__(self) -> str:
        return f"OpenCALM: {self.model_id}"

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation],
    ) -> Iterator[str]:
        model = AutoModelForCausalLM.from_pretrained(
            f"cyberagent/{self.model_id}", device_map="auto", torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(f"cyberagent/{self.model_id}")
        inputs = tokenizer(prompt.prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
        return [output]
