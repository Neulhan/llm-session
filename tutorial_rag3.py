from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-4o-mini")

loader = WebBaseLoader(
    web_paths=("https://www.hyundai.co.kr/story/CONT0000000000163479",),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

# 대표적인 RAG 프롬프트 템플릿을 한국어로 번역한 프롬프트
# 영어 버전으로 사용하는게 LLM 입장에서는 알아듣기 편하나, 성능 낮은 모델의 경우 답변이 영어로 나올 수 있음.
prompt = ChatPromptTemplate.from_template("""
당신은 질문 답변 작업의 보조자입니다.
검색된 다음 문맥을 사용하여 질문에 답하세요.
답을 모른다면 모른다고 말하세요.
답변은 최대 세 문장으로 간결하게 작성하세요.

Question: %질문들어갈자리%

Context: {context}

Answer: """)

# 딕셔너리 형태로 체이닝을 시켜도 Dictionary 의 Value 를 invoke 해준다.
chain = {"context": retriever, } | prompt

result = chain.invoke("2024년 한국시리즈 우승팀은?")

print(result)
