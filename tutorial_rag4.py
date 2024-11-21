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

retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_template("""
당신은 질문 답변 작업의 보조자입니다.
검색된 다음 문맥을 사용하여 질문에 답하세요.
답을 모른다면 모른다고 말하세요.
답변은 최대 세 문장으로 간결하게 작성하세요.

Question: %질문들어갈자리%

Context: {context}

Answer: """)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# invoke 가 안 되는 일반 함수도 아래처럼 체이닝이 가능하다. 이를 RunnableLambda 라고 부른다
chain = {"context": retriever | format_docs} | prompt

result = chain.invoke("2024년 한국시리즈 우승팀은?")

print(result)
