from langchain_community.llms import Ollama

class OllamaGateway:
    def __init__(self):
        system_prompt = ""
        self.llm = Ollama(model="mistral", system=system_prompt)

    def get_response(self, prompt):
        return self.llm(prompt)
