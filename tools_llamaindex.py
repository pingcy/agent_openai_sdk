from llama_index.tools.tavily_research import TavilyToolSpec
import os
from agents import function_tool

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_parse import LlamaParse
from llama_index.core.tools import QueryEngineTool
from llama_index.tools.tavily_research.base import TavilyToolSpec
from llama_index.core import PromptTemplate
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)

my_qa_prompt_tmpl_str = (
    "以下是上下文信息。\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "仅根据上下文信息详细的回答问题，不要依赖于预置知识，不要编造。如果上下文信息无法解答问题，请回答'无法查询到必要的信息'。\n"
    "问题: {query_str}\n"
    "回答: "
)

my_qa_prompt_tmpl = PromptTemplate(my_qa_prompt_tmpl_str)

llm_openai=OpenAI(model="gpt-4o-mini")
llm_embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 默认模型
Settings.llm = llm_openai
Settings.embed_model = llm_embed_model

CHROMA_DIR = "./chroma_data"
RAG_FILE = "./DeepSeek-R1-zh.pdf"

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("deepseek_docs")
vector_store = ChromaVectorStore(chroma_collection=collection)

def create_query_engine(file):

    # Extract filename without extension to use as name
    name = os.path.splitext(os.path.basename(file))[0]

    print(f'Starting to create query engine for 【{name}】...\n')

    if not os.path.exists(f"./storage_chroma/{name}"):
        print('Creating vector index...\n')
        storage_context =  StorageContext.from_defaults(vector_store=vector_store)
        documents = LlamaParse(language='ch_sim',result_type="markdown",verbose=True).load_data(file)
        vector_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        vector_index.storage_context.persist(persist_dir=f"./storage_chroma/{name}")
    else:
        print('Loading vector index...\n')
        storage_context =  StorageContext.from_defaults(persist_dir=f"./storage_chroma/{name}",vector_store=vector_store)
        vector_index = load_index_from_storage(storage_context=storage_context)

    query_engine = vector_index.as_query_engine(similarity_top_k=5,llm=llm_openai)
    query_engine.update_prompts({"response_synthesizer:text_qa_template": my_qa_prompt_tmpl})
    
    return query_engine

# 全局变量
_query_engine = None

def get_query_engine():
    """
    获取 query_engine 单例
    """
    global _query_engine
    if _query_engine is None:
        _query_engine = create_query_engine(RAG_FILE)
    return _query_engine

@function_tool
def rag_query(query_str:str):
    """
    从Deepseek文档查询Deepseek技术细节问题
    
    Args:
        query_str: 查询问题

    Returns:
        result_str: 查询的结果
    """
    print(f'Start rag search {query_str}...')
    result = get_query_engine().query(query_str)
    return str(result)

from agents import function_tool,RunContextWrapper
from dataclasses import dataclass
@dataclass
class UserInfo:
    UserId: str
    UserName: str

# 搜索
@function_tool
def search_web(wrapper: RunContextWrapper[UserInfo],query_str:str):
    """
    使用Tavily进行网络搜索并返回结果
    
    Args:
        query_str: 搜索关键词
    
    Returns:
        result_str: 搜索的相关结果
    """

    print(f'Start web search for 【{wrapper.context.UserName}】 with {query_str}...')
    searh_tool = TavilyToolSpec(os.getenv('TAVILY_API_KEY'))
    search_results = searh_tool.search(query_str,max_results=3)
    return "\n\n".join([result.text for result in search_results])

