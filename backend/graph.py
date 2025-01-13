import contextlib
import os
import uuid
import json
import ast
import time
import requests
from collections import defaultdict
from typing import Annotated, Iterator, Literal, Optional, Sequence, TypedDict, Dict, Any, List, Union

import weaviate
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField, RunnableConfig, ensure_config
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import END, StateGraph, add_messages
from langsmith import Client as LangsmithClient
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools import TavilySearchResults
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManager, RunManager

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.ingest import get_embeddings_model
from backend.stock_utils import extract_and_fetch_stock_data, format_stock_info


# RESPONSE_TEMPLATE = """
# You are an expert in stocks, finance, and cryptocurrencies, tasked with answering any question related to these domains. You can communicate fluently in both English and Chinese.

# Generate a comprehensive and informative answer of 500 words or less for the given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.

# Strictly maintain the length limit of 500 words/characters.

# IMPORTANT: 
# - For Chinese responses: Your answer MUST NOT exceed 500 characters. Keep responses focused and informative.

# You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.

# If there is nothing in the context relevant to the question at hand, try your best to create an answer based on your own knowledge as a stock, finance, and crypto expert.

# If the input question is in Chinese, respond in Chinese. If the input question is in English, respond in English.

# Anything between the following context html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

# <context>

# {context}

# <context/>

# REMEMBER: If there is no relevant information within the context, create an answer based on your own knowledge as a stock, finance, and crypto expert and include the note about the source of the information. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

# 如果输入的问题是中文,请用中文回答。如果输入的问题是英文,请用英文回答。
# """

RESPONSE_TEMPLATE = """
You are an expert in stocks, finance, and cryptocurrencies, tasked with answering any question related to these domains. You can communicate fluently in both English and Chinese.

CRYPTOCURRENCY PRICE QUERIES:
For ANY questions related to cryptocurrencies that involve:
- Current prices or price movements
- Market capitalization
- Trading volume
- Price predictions
- Trading pairs
- Market trends
- Token metrics
You MUST use the most current data. This applies to ALL types of cryptocurrencies, tokens, and digital assets including but not limited to:
- Major cryptocurrencies (Bitcoin, Ethereum, etc.)
- Altcoins and DeFi tokens
- Meme coins and NFT projects
- New or emerging cryptocurrencies

Generate a comprehensive and informative answer of 500 words or less for the given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.

Strictly maintain the length limit of 500 words/characters.

IMPORTANT: 
- For Chinese responses: Your answer MUST NOT exceed 500 characters. Keep responses focused and informative.
- For cryptocurrency price data: Always note the timestamp of the data
- If price data is more than 10 minutes old, warn the user
- Verify data sources are reliable (prefer well-known crypto data aggregators)
- Include relevant market context when available

You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, try your best to create an answer based on your own knowledge as a stock, finance, and crypto expert.

If the input question is in Chinese, respond in Chinese. If the input question is in English, respond in English.

Anything between the following context html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

<context>

{context}

<context/>

REMEMBER: 
- If there is no relevant information within the context, create an answer based on your own knowledge as a stock, finance, and crypto expert and include the note about the source of the information. 
- For cryptocurrency queries requiring current market data, ALWAYS use web search results
- Never use historical knowledge for crypto prices
- Include timestamp for any market data provided
- Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

如果输入的问题是中文,请用中文回答。如果输入的问题是英文,请用英文回答。
"""

COHERE_RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


OPENAI_MODEL_KEY = "openai_gpt_4o_mini"
ANTHROPIC_MODEL_KEY = "anthropic_claude_3_haiku"
FIREWORKS_MIXTRAL_MODEL_KEY = "fireworks_mixtral"
GOOGLE_MODEL_KEY = "google_gemini_pro"
COHERE_MODEL_KEY = "cohere_command"
GROQ_LLAMA_3_MODEL_KEY = "groq_llama_3"
# Not exposed in the UI
GPT_4O_MODEL_KEY = "openai_gpt_4o"
CLAUDE_35_SONNET_MODEL_KEY = "anthropic_claude_3_5_sonnet"

FEEDBACK_KEYS = ["user_score", "user_click"]

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "not_provided")

def update_documents(
    _: list[Document], right: list[Document] | list[dict]
) -> list[Document]:
    res: list[Document] = []

    for item in right:
        if isinstance(item, dict):
            res.append(Document(**item))
        elif isinstance(item, Document):
            res.append(item)
        else:
            raise TypeError(f"Got unknown document type '{type(item)}'")
    return res


class AgentState(TypedDict):
    query: str
    documents: Annotated[list[Document], update_documents]
    messages: Annotated[list[AnyMessage], add_messages]
    # for convenience in evaluations
    answer: str
    feedback_urls: dict[str, list[str]]
    stock_data: Optional[list[dict]]  # New field


gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, streaming=True)

claude_3_haiku = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=4096,
    anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
)
fireworks_mixtral = ChatFireworks(
    model="accounts/fireworks/models/mixtral-8x7b-instruct",
    temperature=0,
    max_tokens=16384,
    fireworks_api_key=os.environ.get("FIREWORKS_API_KEY", "not_provided"),
)
gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    max_output_tokens=16384,
    convert_system_message_to_human=True,
    google_api_key=os.environ.get("GOOGLE_API_KEY", "not_provided"),
)
cohere_command = ChatCohere(
    model="command",
    temperature=0,
    cohere_api_key=os.environ.get("COHERE_API_KEY", "not_provided"),
)
groq_llama3 = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY", "not_provided"),
)

# Not exposed in the UI
gpt_4o = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.3, streaming=True)
claude_35_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.7,
)

llm = gpt_4o.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="model_name"),
    default_key=GPT_4O_MODEL_KEY,
    **{
        ANTHROPIC_MODEL_KEY: claude_3_haiku,
        FIREWORKS_MIXTRAL_MODEL_KEY: fireworks_mixtral,
        GOOGLE_MODEL_KEY: gemini_pro,
        COHERE_MODEL_KEY: cohere_command,
        GROQ_LLAMA_3_MODEL_KEY: groq_llama3,
        OPENAI_MODEL_KEY: gpt_4o_mini,
        CLAUDE_35_SONNET_MODEL_KEY: claude_35_sonnet,
    },
).with_fallbacks(
    [
        gpt_4o,
        gpt_4o_mini,
        claude_3_haiku,
        fireworks_mixtral,
        gemini_pro,
        cohere_command,
        groq_llama3,
    ]
)


#for realtime stock info section
def check_stock_symbols(state: AgentState) -> AgentState:
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    stock_data = extract_and_fetch_stock_data(query)
    if stock_data:
        state["stock_data"] = stock_data
    return state



@contextlib.contextmanager
def get_retriever(k: Optional[int] = None) -> Iterator[BaseRetriever]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ.get("WEAVIATE_API_KEY", "not_provided")
        ),
        skip_init_checks=True,
    ) as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=get_embeddings_model(),
            attributes=["source", "title"],
        )
        k = k or 6
        yield store.as_retriever(search_kwargs=dict(k=k))


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)

def log_event(callbacks, message, metadata=None):
    """统一的日志记录函数"""
    if callbacks:
        # 处理单个 callback manager 的情况
        if hasattr(callbacks, 'on_text'):
            callbacks.on_text(f"\n{message}")
            if metadata:
                callbacks.on_text(f"\nMetadata: {json.dumps(metadata, indent=2)}")
        # 处理 callback 列表的情况
        elif isinstance(callbacks, list) and callbacks:
            run_manager = callbacks[0]
            if hasattr(run_manager, 'on_text'):
                run_manager.on_text(f"\n{message}")
                if metadata:
                    run_manager.on_text(f"\nMetadata: {json.dumps(metadata, indent=2)}")
        # 打印到控制台作为后备选项
        else:
            print(f"\n{message}")
            if metadata:
                print(f"Metadata: {json.dumps(metadata, indent=2)}")

def get_crypto_data(run_manager: Optional[RunManager] = None):
    """获取加密货币数据并创建文档"""
    try:
        log_event(run_manager, "🚀 Starting Binance API request")
        start_time = time.time()
        
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        request_time = time.time() - start_time
        log_event(run_manager, "📡 API request completed", {
            "status_code": response.status_code,
            "request_time": f"{request_time:.2f}s"
        })

        data = response.json()
        web_documents = []
        usdt_pairs = 0
        
        log_event(run_manager, "🔄 Processing cryptocurrency data")
        for item in data:
            if item['symbol'].endswith('USDT'):
                usdt_pairs += 1
                content = (
                    f"Symbol: {item['symbol']}\n"
                    f"Price: ${float(item['lastPrice']):.2f}\n"
                    f"24h Change: {float(item['priceChangePercent'])}%"
                )
                
                display_symbol = item['symbol'].replace('USDT', '')
                
                web_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": "binance",
                            "symbol": display_symbol,
                            "price": float(item['lastPrice']),
                            "volume": float(item['volume'])
                        }
                    )
                )
        
        web_documents.sort(key=lambda x: x.metadata['volume'], reverse=True)
        
        log_event(run_manager, "✅ Data processing completed", {
            "total_pairs": len(data),
            "usdt_pairs": usdt_pairs,
            "processed_pairs": len(web_documents)
        })
        
        return web_documents

    except Exception as e:
        log_event(run_manager, "❌ Error fetching Binance data", {
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        return None
        
def retrieve_documents(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    # 获取 callbacks 用于记录日志
    callbacks = config.get("callbacks") if config else None
    
    config = ensure_config(config)
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    
    log_event(run_manager, "🔍 Starting document retrieval", {
        "query": query
    })
    
    state["query"] = query

    def is_crypto_price_query(query: str) -> bool:
        prompt = """
        Determine if this query is about cryptocurrency prices, market data, or trading information.
        Query: {query}
        Answer with only 'yes' or 'no'.
        """.format(query=query)
        
        log_event(run_manager, "🤔 Checking if query is crypto related")
        response = llm.invoke(prompt).content.lower().strip()
        log_event(run_manager, f"✍️ Query classification result: {response}")
        return 'yes' in response

    if is_crypto_price_query(query):
        log_event(run_manager, "💰 Cryptocurrency price query detected")
        retry_count = 3
        crypto_documents = None
        
        for attempt in range(retry_count):
            try:
                log_event(run_manager, f"🔄 Attempt {attempt + 1} of {retry_count}", {
                    "attempt": attempt + 1,
                    "total_attempts": retry_count
                })
                
                start_time = time.time()
                crypto_documents = get_crypto_data()
                attempt_time = time.time() - start_time
                
                if crypto_documents and len(crypto_documents) > 0:
                    log_event(run_manager, "✅ Successfully retrieved crypto data", {
                        "document_count": len(crypto_documents),
                        "attempt": attempt + 1,
                        "time_taken": f"{attempt_time:.2f}s"
                    })
                    state["documents"] = crypto_documents
                    return state
                else:
                    log_event(run_manager, f"⚠️ No data returned from Binance API", {
                        "attempt": attempt + 1
                    })
                    
            except Exception as e:
                log_event(run_manager, "❌ Error fetching Binance data", {
                    "attempt": attempt + 1,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
            
            if attempt < retry_count - 1:
                wait_time = 5
                log_event(run_manager, "⏳ Waiting before retry", {
                    "wait_time": f"{wait_time}s"
                })
                time.sleep(wait_time)
        
        log_event(run_manager, "🔄 Falling back to web search")
        tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )

        enhanced_query = f"latest {query} cryptocurrency price market data real time"
        log_event(run_manager, "🔍 Enhanced search query", {
            "original_query": query,
            "enhanced_query": enhanced_query
        })
        
        tool_call = {
            "args": {"query": enhanced_query},
            "id": str(uuid.uuid4()),
            "name": "tavily_search",
            "type": "tool_call"
        }
        
        log_event(run_manager, "🌐 Executing web search")
        tool_message = tool.invoke(tool_call)
        search_response = {k: str(v) for k, v in tool_message.artifact.items()}
        results = eval(search_response['results'])
        
        log_event(run_manager, "📝 Processing search results", {
            "result_count": len(results)
        })
        
        web_documents = []
        for result in results:
            content = f"Content: {result['content']}\n"
            if result.get('url'):
                content += f"URL: {result['url']}\n"
                
            web_documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": result.get('url', ''),
                        "title": result.get('title', ''),
                        "type": "web_search",
                        "url": result.get('url', ''),
                    }
                )
            )
        
        if web_documents:
            log_event(run_manager, "✅ Web search successful", {
                "document_count": len(web_documents)
            })
            state["documents"] = web_documents
        else:
            log_event(run_manager, "⚠️ No results found from web search")
            state["documents"] = []
        
        return state
    
    log_event(run_manager, "📚 Using local retriever")
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        relevant_documents = retriever.invoke(query)
        
        should_use_web_search = False
        
        if not relevant_documents:
            should_use_web_search = True
            log_event(run_manager, "⚠️ No documents found in local retriever")
        else:
            log_event(run_manager, "🔍 Checking document relevance")
            relevant_count = 0
            high_quality_docs = []
            
            for doc in relevant_documents:
                score = getattr(doc, 'score', None)
                if score and score < 0.7:
                    continue
                
                content = doc.page_content.strip()
                
                if len(content) < 100:
                    continue
                
                query_words = set(query.lower().split())
                content_words = set(content.lower().split())
                word_overlap = len(query_words.intersection(content_words))
                
                if word_overlap / len(query_words) < 0.3:
                    continue
                
                high_quality_docs.append(doc)
                relevant_count += 1
            
            log_event(run_manager, "📊 Document quality check complete", {
                "total_documents": len(relevant_documents),
                "high_quality_documents": relevant_count
            })
            
            if relevant_count < 2:
                should_use_web_search = True
                log_event(run_manager, "⚠️ Insufficient high-quality documents")
            else:
                relevant_documents = high_quality_docs
        
        if should_use_web_search:
            log_event(run_manager, "🌐 Falling back to web search")
            tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=True,
            )

            tool_call = {
                "args": {"query": query},
                "id": str(uuid.uuid4()),
                "name": "tavily_search",
                "type": "tool_call"
            }
            
            log_event(run_manager, "🔍 Executing web search")
            tool_message = tool.invoke(tool_call)
            search_response = {k: str(v) for k, v in tool_message.artifact.items()}
            results = eval(search_response['results'])

            web_documents = []
            for result in results:
                content = ""
                if result.get('title'):
                    content += f"Title: {result['title']}\n"
                if result.get('content'):
                    content += f"Content: {result['content']}\n"
                if result.get('url'):
                    content += f"URL: {result['url']}\n"
                    
                web_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": result.get('url', ''),
                            "title": result.get('title', ''),
                            "type": "web_search",
                            "url": result.get('url', ''),
                        }
                    )
                )
            
            if web_documents:
                log_event(run_manager, "✅ Web search successful", {
                    "document_count": len(web_documents)
                })
                state["documents"] = web_documents
            else:
                log_event(run_manager, "⚠️ No results found from web search")
                state["documents"] = []
                    
        else:
            log_event(run_manager, "✅ Using local retriever results", {
                "document_count": len(relevant_documents)
            })
            state["documents"] = relevant_documents
    
    return state

def retrieve_documents_with_chat_history(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    config = ensure_config(config)
    model = llm.with_config(tags=["nostream"])
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | model | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    
    # 设置查询到状态中
    state["query"] = query

         # 添加加密货币价格查询检测
    def is_crypto_price_query(query: str) -> bool:
        # 使用 LLM 判断是否是加密货币价格查询
        prompt = """
        Determine if this query is about cryptocurrency prices, market data, or trading information.
        Query: {query}
        Answer with only 'yes' or 'no'.
        """.format(query=query)
        
        response = llm.invoke(prompt).content.lower().strip()
        return 'yes' in response

        # 如果是加密货币价格查询，直接使用 web search
    if is_crypto_price_query(query):
        print("Cryptocurrency price query detected, fetching real-time data...")
        crypto_documents = get_crypto_data()

        if crypto_documents:
            print("Successfully retrieved real-time cryptocurrency data")
            state["documents"] = crypto_documents
            return state
        else:
            print("Cryptocurrency price query detected, using web search...")
            tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=True,
            )
    
            # 优化搜索查询以获取最新价格数据
            enhanced_query = f"latest {query} cryptocurrency price market data real time"
            
            tool_call = {
                "args": {"query": enhanced_query},
                "id": str(uuid.uuid4()),
                "name": "tavily_search",
                "type": "tool_call"
            }
            
            tool_message = tool.invoke(tool_call)
            search_response = {k: str(v) for k, v in tool_message.artifact.items()}
            results = eval(search_response['results'])
            
            web_documents = []
            for result in results:
                content = f"Content: {result['content']}\n"
                if result.get('url'):
                    content += f"URL: {result['url']}\n"
                    
                web_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": result.get('url', ''),
                            "title": result.get('title', ''),
                            "type": "web_search",
                            "url": result.get('url', ''),
                        }
                    )
                )
            
            state["documents"] = web_documents
            return state
    
    # 先尝试本地检索
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        # 获取独立问题
        standalone_question = condense_question_chain.invoke(
            {"question": query, "chat_history": get_chat_history(messages[:-1])}
        )
        print(f"Standalone question: {standalone_question}")
        
        # 直接使用独立问题进行检索
        relevant_documents = retriever.invoke(standalone_question)
        
        # 检查文档相关性
        should_use_web_search = False
        
        if not relevant_documents:
            should_use_web_search = True
            print("No documents found in local retriever")
        else:
            # 使用更严格的相关性检查
            relevant_count = 0
            high_quality_docs = []
            
            for doc in relevant_documents:
                # 相关性评分检查
                score = getattr(doc, 'score', None)
                if score and score < 0.8:  # 提高相关性阈值
                    continue
                
                # 内容质量检查
                content = doc.page_content.strip()
                
                # 检查文档内容长度
                if len(content) < 150:
                    continue
                
                # 检查与独立问题的关键词匹配度
                query_words = set(standalone_question.lower().split())
                content_words = set(content.lower().split())
                word_overlap = len(query_words.intersection(content_words))
                
                # 要求至少有30%的查询关键词出现在文档中
                if word_overlap / len(query_words) < 0.4:
                    continue
                
                # 通过所有检查的文档被认为是高质量的
                high_quality_docs.append(doc)
                relevant_count += 1
            
            # 提高所需的相关文档数量到2
            if relevant_count < 2:
                should_use_web_search = True
                print(f"Only found {relevant_count} relevant documents, not enough. Trying web search...")
            else:
                # 只保留高质量文档
                relevant_documents = high_quality_docs
        
         # 如果需要使用web搜索
        if should_use_web_search:
            print("Using web search for better results...")
            tool = TavilySearchResults(
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=True,
                include_images=True,
            )

            tool_call = {
                "args": {"query": query},
                "id": str(uuid.uuid4()),
                "name": "tavily_search",
                "type": "tool_call"
            }
            
            # 获取 ToolMessage 对象并解析内容
            tool_message = tool.invoke(tool_call)
            search_response = {k: str(v) for k, v in tool_message.artifact.items()}
            results= eval(search_response['results'])

            web_documents = []
            for result in results:
                content = ""
                if result.get('title'):
                    content += f"Title: {result['title']}\n"
                if result.get('content'):
                    content += f"Content: {result['content']}\n"
                if result.get('url'):
                    content += f"URL: {result['url']}\n"
                    
                web_documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": result.get('url', ''),
                            "title": result.get('title', ''),
                            "type": "web_search",
                            "url": result.get('url', ''),
                        }
                    )
                )
            
            if web_documents:
                print("Found results from web search")
                state["documents"] = web_documents
            else:
                print("No results found from web search")
                state["documents"] = []
                    
        else:
            print(f"Found {len(relevant_documents)} high-quality relevant documents in local retriever")
            state["documents"] = relevant_documents
    
    return state

def route_to_retriever(
    state: AgentState,
) -> Literal["retriever", "retriever_with_chat_history"]:
    messages = convert_to_messages(state["messages"])
    
    # 如果已经有文档且为空，说明本地检索失败，使用 web_search
    # if "documents" in state and (not state["documents"] or len(state["documents"]) == 0):
    #     print("No local documents found, routing to web_search")
    #     return "web_search"
    
    # 根据消息长度决定检索方式
    if len(messages) == 1:
        return "retriever"
    else:
        return "retriever_with_chat_history"


def get_chat_history(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    chat_history = []
    for message in messages:
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(
            message, HumanMessage
        ):
            chat_history.append({"content": message.content, "role": message.type})
    return chat_history


def get_feedback_urls(config: RunnableConfig) -> dict[str, list[str]]:
    ls_client = LangsmithClient()
    run_id = config["configurable"].get("run_id")
    if run_id is None:
        return {}

    tokens = ls_client.create_presigned_feedback_tokens(run_id, FEEDBACK_KEYS)
    key_to_token_urls = defaultdict(list)

    for token_idx, token in enumerate(tokens):
        key_idx = token_idx % len(FEEDBACK_KEYS)
        key = FEEDBACK_KEYS[key_idx]
        key_to_token_urls[key].append(token.url)
    return key_to_token_urls



def synthesize_response_old(
    state: AgentState,
    config: RunnableConfig,
    model: LanguageModelLike,
    prompt_template: str,
) -> AgentState:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    response_synthesizer = prompt | model
    synthesized_response = response_synthesizer.invoke(
        {
            "question": state["query"],
            "context": format_docs(state["documents"]),
            # NOTE: we're ignoring the last message here, as it's going to contain the most recent
            # query and we don't want that to be included in the chat history
            "chat_history": get_chat_history(
                convert_to_messages(state["messages"][:-1])
            ),
        }
    )
    # finally, add feedback URLs so that users can leave feedback
    feedback_urls = get_feedback_urls(config)
    return {
        "messages": [synthesized_response],
        "answer": synthesized_response.content,
        "feedback_urls": feedback_urls,
    }

# def synthesize_response(
#     state: AgentState,
#     config: RunnableConfig,
#     model: LanguageModelLike,
#     prompt_template: str,
# ) -> AgentState:
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", prompt_template),
#         ("placeholder", "{chat_history}"),
#         ("human", "{question}"),
#     ])
#     response_synthesizer = prompt | model

#     # 先创建基础 context
#     context = format_docs(state["documents"])
    
#     # 检查是否是 web search 结果
#     if state.get("documents") and state["documents"][0].metadata.get("source") == "web_search":
#        context = "Information from web search:\n" + context
    
#     # 添加股票数据
#     if "stock_data" in state and state["stock_data"]:
#         stock_info = format_stock_info(state["stock_data"])
#         context = stock_info + "\n" + context

#     synthesized_response = response_synthesizer.invoke(
#         {
#             "question": state["query"],
#             "context": context,
#             "chat_history": get_chat_history(
#                 convert_to_messages(state["messages"][:-1])
#             ),
#         }
#     )
#     feedback_urls = get_feedback_urls(config)
#     return {
#         "messages": [synthesized_response],
#         "answer": synthesized_response.content,
#         "feedback_urls": feedback_urls,
#         "query": state["query"],               # 保持状态
#         "documents": state["documents"],       # 保持状态
#         "stock_data": state.get("stock_data")  # 保持状态
#     }

def synthesize_response(
    state: AgentState,
    config: RunnableConfig,
    model: LanguageModelLike,
    prompt_template: str,
) -> AgentState:
    # 检查是否有文档
    has_documents = bool(state.get("documents"))
    
    # 修改 prompt 来处理不同情况
    modified_prompt = prompt_template
    if has_documents:
        # 如果有文档，添加覆盖指令
        modified_prompt += """
        IMPORTANT OVERRIDE: 
        - Relevant information has been found and provided in the context above
        - You MUST use ONLY the information from the provided context
        - Do NOT add the note about AI knowledge
        - Do NOT use your own knowledge
        """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", modified_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    response_synthesizer = prompt | model

    # 创建基础 context
    context = format_docs(state["documents"])
    
    # 检查是否是 web search 结果
    if state.get("documents"):
        is_web_search = any(
            doc.metadata.get("type") == "web_search" 
            for doc in state["documents"]
        )
        if is_web_search:
            context = "Information from web search:\n" + context
    
    # 添加股票数据
    if "stock_data" in state and state["stock_data"]:
        stock_info = format_stock_info(state["stock_data"])
        context = stock_info + "\n" + context

    synthesized_response = response_synthesizer.invoke(
        {
            "question": state["query"],
            "context": context,
            "chat_history": get_chat_history(
                convert_to_messages(state["messages"][:-1])
            ),
        }
    )
    
    feedback_urls = get_feedback_urls(config)
    return {
        "messages": [synthesized_response],
        "answer": synthesized_response.content,
        "feedback_urls": feedback_urls,
        "query": state["query"],               # 保持状态
        "documents": state["documents"],       # 保持状态
        "stock_data": state.get("stock_data")  # 保持状态
    }

def synthesize_response_default(
    state: AgentState, config: RunnableConfig
) -> AgentState:
    return synthesize_response(state, config, llm, RESPONSE_TEMPLATE)


def synthesize_response_cohere(state: AgentState, config: RunnableConfig) -> AgentState:
    model = llm.bind(documents=state["documents"])
    return synthesize_response(state, config, model, COHERE_RESPONSE_TEMPLATE)


def route_to_response_synthesizer(
    state: AgentState, config: RunnableConfig
) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:
    model_name = config.get("configurable", {}).get("model_name", GPT_4O_MODEL_KEY)
    if model_name == COHERE_MODEL_KEY:
        return "response_synthesizer_cohere"
    else:
        return "response_synthesizer"

# 添加新的搜索引擎节点函数
# def web_search_documents(state: AgentState) -> AgentState:

#     messages = convert_to_messages(state["messages"])
#     query = messages[-1].content
#     state["query"] = query

    
#     tool = TavilySearchResults(
#         max_results=5,
#         search_depth="advanced",
#         include_answer=True,
#         include_raw_content=True,
#         include_images=True,
#         # include_domains=[...],
#         # exclude_domains=[...],
#         # name="...",            # overwrite default tool name
#         # description="...",     # overwrite default tool description
#         # args_schema=...,       # overwrite default args_schema: BaseModel
#     )

#     search_results = tool.invoke({"query": query})

#     web_documents = []
#     for result in search_results:
#         content = ""
#         if result.get('content'):
#             content += f"Content: {result['content']}\n"
#         if result.get('url'):
#             content += f"URL: {result['url']}\n"
        
#         web_documents.append(
#             Document(
#                 page_content=content,
#                 metadata={
#                     "source": "web_search",
#                     "title": result.get('title', ''),
#                     "url": result.get('url', ''),
#                 }
#             )
#         )
    
#     state["documents"] = web_documents
#     return state
    


class Configuration(TypedDict):
    model_name: str
    k: int


class InputSchema(TypedDict):
    messages: list[AnyMessage]


workflow = StateGraph(AgentState, Configuration, input=InputSchema)

# define nodes
workflow.add_node("stock_symbol_check", check_stock_symbols)
# workflow.add_node("web_search", web_search_documents)
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("retriever_with_chat_history", retrieve_documents_with_chat_history)
workflow.add_node("response_synthesizer", synthesize_response_default)
workflow.add_node("response_synthesizer_cohere", synthesize_response_cohere)

# set entry point to stock symbol check
workflow.set_entry_point("stock_symbol_check")

# connect stock symbol check to retrievers
workflow.add_conditional_edges(
    "stock_symbol_check",
    route_to_retriever,
    {
        "retriever": "retriever",
        "retriever_with_chat_history": "retriever_with_chat_history"
    }
)

# 连接检索器到响应合成器
workflow.add_conditional_edges(
    "retriever",
    route_to_response_synthesizer,
    {
        "response_synthesizer": "response_synthesizer",
        "response_synthesizer_cohere": "response_synthesizer_cohere"
    }
)
workflow.add_conditional_edges(
    "retriever_with_chat_history",
    route_to_response_synthesizer,
    {
        "response_synthesizer": "response_synthesizer",
        "response_synthesizer_cohere": "response_synthesizer_cohere"
    }
)
# workflow.add_conditional_edges(
#     "web_search",
#     route_to_response_synthesizer,
#     {
#         "response_synthesizer": "response_synthesizer",
#         "response_synthesizer_cohere": "response_synthesizer_cohere"
#     }
# )

# connect synthesizers to terminal node
workflow.add_edge("response_synthesizer", END)
workflow.add_edge("response_synthesizer_cohere", END)

graph = workflow.compile()
