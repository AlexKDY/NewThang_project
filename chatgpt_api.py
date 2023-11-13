import openai
import os
from openai import OpenAI

client = OpenAI()

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatBot:
    
    def __init__(self, model = "gpt-3.5-turbo"):
    
        self.model = model
        self.messages = [{"role": "system", "content": "You are a helpful assistant using korean"}]
        
    def ask(self, question):
    
        self.messages.append({
            'role': 'user', 
            'content': f"{question}"
        })
        response = self.__ask__()
        
        return response
        
    def __ask__(self):
        completion = client.chat.completions.create(
            model = self.model,
            messages = self.messages,
        )
        response = completion.choices[0].message.content
        
        self.messages.append({
            'role': 'assistant', 
            'content': f"{response}"
        })
        return response
    
    def show_messages(self):
        return self.messages
    
    def clear(self):
        self.messages.clear()

class Summarizer:
    
    def __init__(self, model = "gpt-3.5-turbo"):
    
        self.model = model
        self.messages = [{"role": "system", "content": "You are a news summarizaer using korean"}]
        
    def summarize(self, text):
        
        self.messages.append({
            'role': 'user', 
            'content': f"이 기사의 요점을 요약해줘: {text}"
        })
        gist = self.__summary__()

        self.messages.append({
            'role': 'user', 
            'content': f"이 기사의 주요 논거를 요약해줘: {gist}"
        })
        aug = self.__summary__()
        

        self.messages.append({
            'role': 'user', 
            'content': f"이 기사를 3 ~ 4문장으로 요약해줘: {aug}"
        })
        response = self.__summary__()
        
        return response
        
    def __summary__(self):
        completion = client.chat.completions.create(
            model = self.model,
            messages = self.messages,
        )
        response = completion.choices[0].message.content

        return response
    
    def show_messages(self):
        return self.messages
    
    def clear(self):
        self.messages.clear()

class KeywordExtractor:
    
    def __init__(self, model = "gpt-3.5-turbo"):

        self.model = model
        self.messages = [{"role": "system", "content": "You are a keyword extractor"}]
        
    def extract(self, text):
    
        self.messages.append({
            'role': 'user', 
            'content': f"이 기사의 키워드를 3개 추출해줘{text}"
        })
        response = self.__extract__()
        
        return response
        
    def __extract__(self):
        completion = client.chat.completions.create(
            model = self.model,
            messages = self.messages,
        )
        response = completion.choices[0].message.content

        return response
    
    def show_messages(self):
        return self.messages
    
    def clear(self):
        self.messages.clear()


if __name__ == '__main__':
    chatbot = ChatBot()
    summarizer = Summarizer()
    keyword = KeywordExtractor()

    '''
    while True:
        question = input("User: ")
        if question == "Exit":
            print("Chatbot Exit")
            break
    '''
    text = ""
    with open('./dataset/news3.txt', 'r') as f:
        for line in f:
            if len(line) > 0:
                for word in line.split():
                    text += f" {word}"
    
    print(summarizer.summarize(text))

            



    
    