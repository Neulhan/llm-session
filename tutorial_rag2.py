from langchain_openai import ChatOpenAI
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

# split 된 문서들을 ChromaDB 에 embedding 해서 넣어주는 과정
# ChromaDB 는 대표적인 오픈소스 벡터 데이터베이스 중 하나
# 어떤 임베딩모델을 사용할지, 어떤 문서 조각을 저장할지를 정해준다.
# OpenAIEmbeddings 에 아무 인자도 주지 않으면 "text-embedding-ada-002" 모델이 사용된다
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# 임베딩을 저장한 DB 를 리트리버로 만들기.
retriever = vectorstore.as_retriever()

# 리트리버는 invoke 호출 가능한 Runnable 이다. 체이닝이 가능하다는 뜻
docs = retriever.invoke("2024년 한국시리즈 우승팀은?")

for doc in docs:
    print(f"{doc=}", end="\n\n")
