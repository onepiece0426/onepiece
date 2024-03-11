from langchain.llms import ChatGLM


endpoint_url = "http://127.0.0.1:8000"

llms = ChatGLM(
    endpoint_url=endpoint_url,
    max_tonken=80000,
    tpo_p=0.9
)


