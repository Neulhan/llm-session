from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage(content="Translate the following from English into Korean"),
    HumanMessage(content="hi!"),
]

# StrOutputParser 는 결과물에서 Str(문자열)만 뽑아준다
parser = StrOutputParser()


result = llm.invoke(messages)
print(result) # content='안녕!' additional_kwargs={'refusal': None} response_metadata......

result = parser.invoke(result)
print(result) # 안녕!
