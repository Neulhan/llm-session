from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 랭체인에는 이미 만들어져있는 여러가지 Loader 들이 존재한다.
# (PDFLoader, GitLoader, GoogleDriveLoader, NotionDBLoader 등등)
# 그 중에서 WebBaseLoader 는 웹에서 정보를 Load 해오는 Loader
loader = WebBaseLoader(
    web_paths=("https://www.hyundai.co.kr/story/CONT0000000000163479",),
)

# loader 는 .load() 라는 일관된 인터페이스를 가지고 있다.
docs = loader.load()

print(docs)

# 다양한 TextSplitter 중 RecursiveCharacterTextSplitter 는
# "\n\n" > "\n" > " " > "" 순서로 텍스트 분할을 시도하는 Splitter
# 아래는 1,000 글자 단위로 split 하되, 200자 까지 겹치는걸 허용한다는 옵션
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print()
print(splits)
