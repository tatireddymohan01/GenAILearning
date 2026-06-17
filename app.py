from dotenv import load_dotenv
import os

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

print("Environment variables loaded successfully.")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
print(model)

model_response = model.invoke("briefly explain the concept of artificial intelligence in two lines")
print(model_response.content)


from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "{input}"),
        ("ai", "The answer should just name the capital city without any additional information.")
    ]
)

from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()

chain = prompt | model | output_parser
response = chain.invoke({"input":"capital of India?"})
print(response)

from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()
docs = retriever.invoke("What is the capital of India?")
print(docs[0].page_content)

