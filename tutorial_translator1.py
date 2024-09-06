from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# LLM 모델, OpenAI 외에 다른 모델도 유사하게 불러와서 쓰면 된다.
llm = ChatOpenAI(model="gpt-4o")

# messages. LLM 에 들어갈 메세지를 넣어준다.
# SystemMessage 는 대화 바깥의 메세지. 시스템에게 명령을 입력해두는 것
# HumanMessage 는 대화에서 인간 측의 메세지
messages = [
    SystemMessage(content="다음을 영어에서 한국어로 번역하세요."),
    HumanMessage(content="hi!"),
]

# LLM 에 message 를 invoke 하면 결과를 받아볼 수 있다.
result = llm.invoke(messages)


print(result.content)
