import os
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


class ModelManager:
    def __init__(self, model_name: str, token: str, lora_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_auth_token=token,
            load_in_8bit=True,
            device_map="auto"
        )

        if lora_path:
            self.model = PeftModel.from_pretrained(self.model, lora_path)

        self.pipeline = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, max_tokens: int = 200) -> str:
        result = self.pipeline(prompt, max_new_tokens=max_tokens)
        return result[0]['generated_text']


llama_model = ModelManager(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    token=HF_TOKEN,
    lora_path="fine_tuning/lora_output"
)

