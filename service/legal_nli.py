import torch
from gateway.openai import OpenAIGateway
from gateway.ollama import OllamaGateway
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "frisibeli/legal-bert-finetuned-1985-classifier"

class LegalNLIResponse:
    def __init__(self, label, probilities = None):
        self.label = label
        if probilities:
            self.probilities = probilities

    def to_json(self):
        return {
            'label': self.label,
            'probabilities': self.probilities
        }

class LegalNLI:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    def predict(self, hypothesis, premise):
        inputs = self.tokenizer(
            hypothesis,
            premise,
            padding = 'max_length',
            truncation=True,
            max_length = 512,
            return_tensors='pt',
        )

         # Perform inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Predict label
        probabilities = logits.softmax(dim=-1)
        label = torch.argmax(probabilities, dim=-1).item()

        return LegalNLIResponse(label, probabilities.tolist())

    def predict_gpt4(self, explaination, hypothesis, answer):
        openai_gateway = OpenAIGateway()
        system_prompt = """
        You are a legal assistant with a specialization in U.S. Civil Procedure. Your role involves thorough analysis and resolution of cases pertaining to this field. You will encounter three key components in each case:

        1. EXPLANATION: This provides additional context and background information about a specific lawsuit.
        2. QUESTION: Here, you will be presented with actual facts and details surrounding the lawsuit.
        3. HYPOTHESIS: Based on the provided information, a hypothesis will be presented. Your task is to rigorously evaluate this hypothesis in the context of U.S. Civil Procedure and determine its validity.
        Respond ONLY with 'TRUE' if you conclude that the hypothesis is correct, or ONLY with 'FALSE' if you find it to be incorrect.
        Do not provide any reasoning and ONLY answer with 'TRUE' or 'FALSE'. I'm going to tip $300K for the correct solution!
        """

        input_template = """
        EXPLANATION: "{}"

        QUESTION: "{}"

        HYPOTHESIS: "{}"
        """
        prompt = [
            ("system", system_prompt),
            ("human", input_template.format(explaination, hypothesis, answer)),
        ]
        return openai_gateway.get_response_gpt4(prompt)

    def predict_mistral(self, explaination, hypothesis, answer):
        print("Predicting with Mistral")
        ollama_gateway = OllamaGateway()
        prompt = """
        As a proficient civil law assistant, limit your response to "TRUE" or "FALSE". Use "TRUE" when you believe that the STATEMENT
        aligns with the given CONTEXT and CASE based on your understanding. If unsure, opt for "FALSE". The CONTEXT encompasses {} |
        The CASE presents the question as {}. The STATEMENT to be evaluated is: {}.
        """.format(explaination, hypothesis, answer)

        return ollama_gateway.get_response(prompt)
