from gateway.openai import OpenAIGateway
from transformers import pipeline

class Summarization:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0, truncation=True)

    def summarize_gpt4(self, text):
        openai_gateway = OpenAIGateway()
        system_prompt = "You are a Legal AI assistant. For each user input, your sole task is to output a concise summary, up to 400 tokens, of the following legal context from a school book. The context explores essential principles and case law related to US Civil Law, encompassing key concepts, landmark decisions, and their implications. Craft a summary that captures the core ideas and arguments, tailored for a legal audience seeking a comprehensive understanding of the subject matter."

        prompt = [
            ("system", system_prompt),
            ("human", text),
        ]

        return openai_gateway.get_response_gpt4(prompt)

    def summarize_bart(self, text):
        return self.summarizer(text, max_length=400, min_length=50, do_sample=False)[0]['summary_text']
