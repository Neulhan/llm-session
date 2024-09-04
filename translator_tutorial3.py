from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4o")
messages = [
    SystemMessage(content="Translate the following from English into Korean"),
    HumanMessage(content="hi!"),
]

# invoke 를 연쇄적으로 호출하는 경우 LECL(랭체인 표현식 언어) 로 chaining 을 한 다음
chain = model | parser

# 한 번의 invoke 로 처리할 수 있다.
result = chain.invoke(messages)

# 이렇게 invoke 가 가능한 객체를 랭체인에서는 Runnable 이라고 부른다.
print(result)
