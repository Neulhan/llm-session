from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages(
    [("system", "Translate the following into {language}:"), ("user", "{text}")]
)
llm = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

# prompt, llm, parser 세 개의 Runnable 을 체이닝해서 체인으로 만든 상태
chain = prompt | llm | parser

# 입력({"language": "Korean", "text": "Hi!"})이 prompt 로,
# prompt 결과물을 입력으로 써서 LLM 을 호출,
# LLM 결과물을 입력으로 써서 parser 호출
result = chain.invoke({"language": "Korean", "text": "Hi!"})

# result 에는 최종적으로 parser 의 결과물이 나온다.
print(result)
