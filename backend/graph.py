import contextlib
import os
from collections import defaultdict
from typing import Annotated, Iterator, Literal, Optional, Sequence, TypedDict

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

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.ingest import get_embeddings_model
from backend.stock_utils import extract_and_fetch_stock_data, format_stock_info


RESPONSE_TEMPLATE = """
You are an expert in stocks, finance, and cryptocurrencies, tasked with answering any question related to these domains. You can communicate fluently in both English and Chinese.

Generate a comprehensive and informative answer of 300 words or less for the given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.

Strictly maintain the length limit of 300 words/characters.

IMPORTANT: 
- For Chinese responses: Your answer MUST NOT exceed 320 characters. Keep responses focused and informative.

You should use bullet points in your answer for readability. Put citations where they apply rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, try your best to create an answer based on your own knowledge as a stock, finance, and crypto expert, but add a note at the end stating: "Note: This response is based on the AI's own knowledge as a stock, finance, and crypto expert, as no relevant information was found in the provided context."

If the input question is in Chinese, respond in Chinese. If the input question is in English, respond in English.

Anything between the following context html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

<context>

{context}

<context/>

REMEMBER: If there is no relevant information within the context, create an answer based on your own knowledge as a stock, finance, and crypto expert and include the note about the source of the information. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.

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


# def retrieve_documents(
#     state: AgentState, *, config: Optional[RunnableConfig] = None
# ) -> AgentState:
#     config = ensure_config(config)
#     messages = convert_to_messages(state["messages"])
#     query = messages[-1].content
    
#     # 首先设置查询到状态中
#     state["query"] = query
    
#     with get_retriever(k=config["configurable"].get("k")) as retriever:
#         relevant_documents = retriever.invoke(query)
#         # 如果没有找到相关文档，设置为空列表
#         if not relevant_documents:
#             print("No relevant documents found in retriever")
#             state["documents"] = []
#         else:
#             state["documents"] = relevant_documents
    
#     return state

def retrieve_documents(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    config = ensure_config(config)
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    
    # 首先设置查询到状态中
    state["query"] = query
    
    # 先尝试本地检索
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        relevant_documents = retriever.invoke(query)
        # 如果没有找到相关文档，尝试web搜索
        if not relevant_documents:
            print("No relevant documents found in retriever, trying web search...")
                # 创建 Tavily 搜索工具
                tool = TavilySearchResults(
                    max_results=5,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=True,
                    include_images=True,
                )

                # 执行网络搜索
                search_results = tool.invoke({"query": query})

                web_documents = []
                for result in search_results:
                    content = ""
                    if result.get('content'):
                        content += f"Content: {result['content']}\n"
                    if result.get('url'):
                        content += f"URL: {result['url']}\n"
                    
                    web_documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": "web_search",
                                "title": result.get('title', ''),
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
            print("Found relevant documents in local retriever")
            state["documents"] = relevant_documents
    
    return state

# def retrieve_documents_with_chat_history(
#     state: AgentState, *, config: Optional[RunnableConfig] = None
# ) -> AgentState:
#     config = ensure_config(config)
#     condense_question_chain = create_condense_question_chain(
#         llm,
#         "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."
#     )

#     messages = convert_to_messages(state["messages"])
#     query = messages[-1].content
    
#     # 设置查询到状态中
#     state["query"] = query
    
#     with get_retriever(k=config["configurable"].get("k")) as retriever:
#         retriever_with_condensed_question = condense_question_chain | retriever
#         relevant_documents = retriever_with_condensed_question.invoke(
#             {"question": query, "chat_history": get_chat_history(messages[:-1])}
#         )
#         state["documents"] = relevant_documents if relevant_documents else []
    
#     return state

def retrieve_documents_with_chat_history(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    config = ensure_config(config)
    condense_question_chain = create_condense_question_chain(
        llm,
        "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    
    # 设置查询到状态中
    state["query"] = query
    
    # 先尝试本地检索
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        retriever_with_condensed_question = condense_question_chain | retriever
        relevant_documents = retriever_with_condensed_question.invoke(
            {"question": query, "chat_history": get_chat_history(messages[:-1])}
        )
        
        # 如果本地检索没有找到文档，尝试web搜索
        if not relevant_documents:
            print("No relevant documents found in retriever with chat history, trying web search...")
                # 创建 Tavily 搜索工具
                tool = TavilySearchResults(
                    max_results=5,
                    search_depth="advanced",
                    include_answer=True,
                    include_raw_content=True,
                    include_images=True,
                )

                # 使用压缩后的独立问题进行网络搜索
                standalone_question = condense_question_chain.invoke(
                    {"question": query, "chat_history": get_chat_history(messages[:-1])}
                )
                
                print(f"Searching web with standalone question: {standalone_question}")
                search_results = tool.invoke({"query": standalone_question})

                web_documents = []
                for result in search_results:
                    content = ""
                    if result.get('content'):
                        content += f"Content: {result['content']}\n"
                    if result.get('url'):
                        content += f"URL: {result['url']}\n"
                    
                    web_documents.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": "web_search",
                                "title": result.get('title', ''),
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
            print("Found relevant documents in local retriever with chat history")
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

def synthesize_response(
    state: AgentState,
    config: RunnableConfig,
    model: LanguageModelLike,
    prompt_template: str,
) -> AgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_template),
        ("placeholder", "{chat_history}"),
        ("human", "{question}"),
    ])
    response_synthesizer = prompt | model

    # 先创建基础 context
    context = format_docs(state["documents"])
    
    # 检查是否是 web search 结果
    if state.get("documents") and state["documents"][0].metadata.get("source") == "web_search":
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
