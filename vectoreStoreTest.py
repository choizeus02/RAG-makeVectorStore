import os
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough
import faiss
import boto3

load_dotenv()
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import DirectoryLoader

from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings


# 단계 1 문서 로드(Load Documents)

# 단계 1: 문서 로드
loader = DirectoryLoader('최종데이터/test', glob="**/*.json", loader_cls=JSONLoader, loader_kwargs = {'jq_schema':'.', 'text_content':False})
data = loader.load()
OPENAI_API_TOKEN = "sk-proj-"
os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN     # OpenAI API 토큰 설정



print(f"Document Loader 사용시 문서 수 {len(data)}")
# print(data)
# print(f"문서의 수: {len(data)}")

# 단계 2 문서 분할

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)

#  문서 로드 및 분할(load_and_split)
split_doc = text_splitter.split_documents(data)
# for sent in text_splitter.split_documents(data):
#     print(sent)



# 단계 3 임베딩 & 벡터스토어 생성(Create Vectorstore)


# S3 버킷에 FAISS 인덱스를 저장하는 함수
def save_faiss_index_to_s3(index, s3_bucket, s3_key, local_path='faiss_index.index'):
    faiss.write_index(index, local_path)
    s3_client = boto3.client('s3')
    s3_client.upload_file(local_path, s3_bucket, s3_key)
    os.remove(local_path)

# S3 버킷에서 FAISS 인덱스를 불러오는 함수
def load_faiss_index_from_s3(s3_bucket, s3_key, local_path='faiss_index.index'):
    s3_client = boto3.client('s3')
    s3_client.download_file(s3_bucket, s3_key, local_path)
    index = faiss.read_index(local_path)
    os.remove(local_path)
    return index

# 데이터를 로드합니다.
documents = data

# FAISS 인덱스를 생성하고 S3에 저장합니다.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
faiss_vectorstore = FAISS.from_documents(documents, embeddings)
save_faiss_index_to_s3(faiss_vectorstore.index, '', 'path/in/s3/faiss_index1.index')

# S3에서 FAISS 인덱스를 불러옵니다.
faiss_index = load_faiss_index_from_s3('', 'path/in/s3/faiss_index1.index')
# FAISS 벡터 스토어를 초기화합니다.
faiss_vectorstore = FAISS(
    embedding_function=embeddings.embed_query,
    index=faiss_index,
    docstore=faiss_vectorstore.docstore,
    index_to_docstore_id=faiss_vectorstore.index_to_docstore_id
)


# 사용자의 질문(query)에 부합하는 문서를 검색합니다.
k = 3

# BM25 retriever 초기화
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = k

# FAISS retriever 초기화
faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

# Ensemble retriever 초기화
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
)

# RAG 프롬프트 생성
prompt = hub.pull("rlm/rag-prompt")
print(prompt)

# 언어 모델 생성
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 검색한 문서 결과를 하나의 문단으로 합치는 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 체인 생성
rag_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(f"문서의 수: {len(data)}")
print("===" * 20)
question = "라이젠7 5700X + RX 6700 XT 조합에 대해 설명해줘"
response = rag_chain.invoke(question)
print(response)
