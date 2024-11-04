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

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.ingest import get_embeddings_model
from backend.stock_utils import extract_and_fetch_stock_data, format_stock_info


RESPONSE_TEMPLATE = """
You are an expert in stocks, finance, and cryptocurrencies, tasked with answering any question related to these domains. You can communicate fluently in both English and Chinese.

Generate a comprehensive and informative answer of 260 words or less for the given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.

Strictly maintain the length limit of 260 words/characters.

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


def retrieve_documents(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    config = ensure_config(config)
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        relevant_documents = retriever.invoke(query)
    return {"query": query, "documents": relevant_documents}


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
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        retriever_with_condensed_question = condense_question_chain | retriever
        # NOTE: we're ignoring the last message here, as it's going to contain the most recent
        # query and we don't want that to be included in the chat history
        relevant_documents = retriever_with_condensed_question.invoke(
            {"question": query, "chat_history": get_chat_history(messages[:-1])}
        )
    return {"query": query, "documents": relevant_documents}


def route_to_retriever(
    state: AgentState,
) -> Literal["web_search", "retriever", "retriever_with_chat_history"]:
    # 首先尝试本地检索
    with get_retriever() as retriever:
        messages = convert_to_messages(state["messages"])
        query = messages[-1].content
        documents = retriever.invoke(query)
    
    # 如果本地检索没有结果，使用 web_search
    if not documents:
        return "web_search"
    
    # 如果有结果，根据消息长度决定使用哪种检索方式
    if len(state["messages"]) == 1:
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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    response_synthesizer = prompt | model

    if state.get("documents") and state["documents"][0].metadata.get("source") == "web_search":
       context = "Information from web search:\n" + context
    
    # Include stock data in the context if available
    context = format_docs(state["documents"])
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
def web_search_documents(state: AgentState) -> AgentState:
    """当数据库检索无结果时，使用Tavily搜索引擎获取信息"""
    try:
        search = TavilySearchAPIWrapper(
            tavily_api_key=os.environ.get("TAVILY_API_KEY"),
            search_depth="advanced",  # 可以是 "basic" 或 "advanced"
            max_results=5  # 可以调整结果数量
        )
        messages = convert_to_messages(state["messages"])
        query = messages[-1].content
        
        # 执行Tavily搜索
        search_results = search.run(query)
        
        # 将搜索结果转换为Document格式
        web_documents = [
            Document(
                page_content=search_results,
                metadata={
                    "source": "web_search",
                    "title": "Tavily Search Results"
                }
            )
        ]
        
        state["documents"] = web_documents
        print("Tavily Search completed successfully")
        
    except Exception as e:
        print(f"Tavily Search error: {str(e)}")
        # 搜索失败时返回空文档
        state["documents"] = []
    
    return state


class Configuration(TypedDict):
    model_name: str
    k: int


class InputSchema(TypedDict):
    messages: list[AnyMessage]


workflow = StateGraph(AgentState, Configuration, input=InputSchema)

# define nodes
workflow.add_node("stock_symbol_check", check_stock_symbols)
workflow.add_node("web_search", web_search_documents)
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("retriever_with_chat_history", retrieve_documents_with_chat_history)
workflow.add_node("response_synthesizer", synthesize_response_default)
workflow.add_node("response_synthesizer_cohere", synthesize_response_cohere)

# set entry point to stock symbol check
workflow.set_entry_point("stock_symbol_check")

# connect stock symbol check to retrievers
workflow.add_conditional_edges("stock_symbol_check", route_to_retriever)

# connect retrievers and response synthesizers
workflow.add_conditional_edges("retriever", route_to_response_synthesizer)
workflow.add_conditional_edges(
    "retriever_with_chat_history", route_to_response_synthesizer
)
workflow.add_conditional_edges(
    "web_search", 
    route_to_response_synthesizer)

# connect synthesizers to terminal node
workflow.add_edge("response_synthesizer", END)
workflow.add_edge("response_synthesizer_cohere", END)

graph = workflow.compile()
