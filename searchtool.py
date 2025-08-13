import os
from dotenv import load_dotenv
load_dotenv()

USER_AGENT = os.getenv("USER_AGENT")

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import WebBaseLoader
import re

llm = ChatOllama(model="phi4-mini:latest",temperature=0.1)  # Using ChatOllama for summarization
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, )  # Alternative using OpenAI's model


memory = ConversationBufferWindowMemory(
    k=3,
    return_messages=True,
    memory_key="chat_history",
    input_key="input",
)

# Cache for storing latest summary
latest_summary = {"content": ""}

followup_memory = ConversationBufferWindowMemory(
    k=3,
    memory_key="chat_history",
    input_key="question",
    output_key="text",      
    return_messages=False    
)



def summarize_url(url: str) -> str:
    """Load, summarize the URL, cache the summary, and reset follow-up memory."""
    loader = WebBaseLoader(url)
    docs = loader.load()
    summarize_chain = load_summarize_chain(llm, chain_type="stuff")
    summary = summarize_chain.run(docs).strip()
    latest_summary["content"] = summary

    # New context → clear prior follow-up conversation
    followup_memory.clear()
    return summary

# Tool for summarization
def summarize_tool_func(url: str) -> str:
    try:
        return summarize_url(url)
    except Exception as e:
        return f"Error: {str(e)}"

# LangChain Tool
summarization_tool = Tool(
    name="Web Summarizer",
    func=summarize_tool_func,
    description="Use this tool to summarize a webpage from a given URL."
)

# Initialize agent with memory
def get_agent():
    return initialize_agent(
        tools=[summarization_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )



# --- add below existing chains ---
TOPIC_TEMPLATE = """You are given a summary of a webpage.
Write a single, friendly, neutral, and polite TOPIC line in Title Case.
Constraints:
- Maximum 6 words.
- No emojis.
- No trailing punctuation.
- Be specific, not clickbait.

Summary:
{summary}

Topic:"""

topic_prompt = PromptTemplate(
    template=TOPIC_TEMPLATE,
    input_variables=["summary"],
)

topic_chain = LLMChain(llm=llm, prompt=topic_prompt, verbose=False)

def topic_from_summary(summary: str) -> str:
    """Create a concise, polite topic from a summary (<= 6 words)."""
    raw = topic_chain.predict(summary=summary).strip()

    # Keep only the first line
    first_line = raw.splitlines()[0].strip()

    # Remove surrounding quotes and trailing punctuation
    cleaned = re.sub(r'^[\'"“”‘’]|[\'"“”‘’]\s*$', '', first_line).strip()
    cleaned = re.sub(r'[.!?]+$', '', cleaned).strip()

    return cleaned


# ===== Follow-up prompt/chain (answers strictly from cached summary) =====
FOLLOWUP_TEMPLATE = """You can use ONLY the provided summary and the chat_history.
If the answer is not present in the summary, reply exactly: "I don't know based on the summary."
Then suggest summarizing another URL for more context.

chat_history:
{chat_history}

summary:
{summary}

question: {question}
answer:"""

followup_prompt = PromptTemplate(
    template=FOLLOWUP_TEMPLATE,
    input_variables=["chat_history", "summary", "question"],
)

followup_chain = LLMChain(
    llm=llm,
    prompt=followup_prompt,
    memory=followup_memory,
    verbose=False,
)

def answer_followup(question: str) -> str:
    """Answer a follow-up using ONLY the cached summary + the memory window."""
    summary = latest_summary.get("content", "").strip()
    if not summary:
        return "No summary is available yet. Please provide a URL to summarize first."
    return followup_chain.predict(summary=summary, question=question).strip()

