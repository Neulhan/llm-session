from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [("system", "Translate the following into {language}:"),
    ("user", "{text}")]
)

# Prompt 에 자리(language, text) 만들어두고 invoke 하면 해당 자리에 들어간다
result = prompt_template.invoke({"language": "Korean", "text": "hi"})

# invoke 가 되기 때문에 Prompt 도 하나의 Runnable 로 취급되는걸 알 수 있다.
print(result)
