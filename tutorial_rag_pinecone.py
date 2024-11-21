from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# import PineconeVectorStore
from langchain_pinecone import PineconeVectorStore


llm = ChatOpenAI(model="gpt-4o-mini")

loader = WebBaseLoader(
    web_paths=("https://www.hyundai.co.kr/story/CONT0000000000163479",),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 내가 만든 index 의 이름, 사용할 임베딩 모델, 복사해온 api key 를 넣어준다
# 환경변수에 PINECONE_API_KEY 등록했으면 api_key 는 생략해도 됨
vectorstore = PineconeVectorStore(
    index_name="my-first-index",
    embedding=OpenAIEmbeddings(),
)

# Pinecone 에 문서를 추가하는 과정, 이 코드가 실행되고 난 뒤 Pinecone 관리 페이지를 다시 들어가보면
# Documents 가 생성된 것을 확인할 수 있다
vectorstore.add_documents(documents=splits)


# 이후 Retriever 로의 활용 방법은 동일
retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_messages([("user", """
당신은 질문 답변 작업의 보조자입니다. 검색된 다음 문맥을 사용하여 질문에 답하세요. 답을 모른다면 모른다고 말하세요. 답변은 최대 세 문장으로 간결하게 작성하세요.

Question: {question}

Context: {context}

Answer:
""")])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("2024년 한국시리즈 우승팀은?")


print(result)
