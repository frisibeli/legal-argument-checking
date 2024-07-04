from langchain_community.chat_models import ChatOpenAI

class OpenAIGateway:
    def __init__(self):
        self.api_key = ""


    def get_response_gpt4(self, prompt):
        llm = ChatOpenAI(temperature=0.7, openai_api_key=self.api_key, model_name="gpt-4")
        response = llm.invoke(prompt)
        return response.content
