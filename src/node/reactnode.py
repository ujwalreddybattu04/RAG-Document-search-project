"""LangGraph nodes for RAG workflow + ReAct Agent"""

from typing import Optional
from src.state.rag_state import RAGState

from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy init

    # ----------------------------------------------------------
    # RETRIEVER NODE
    # ----------------------------------------------------------
    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
        )

    # ----------------------------------------------------------
    # TOOL BUILDING (NO TYPE HINTS → FIXES PYTHON 3.13 ISSUES)
    # ----------------------------------------------------------
    def _build_tools(self):
        """Create retriever + Wikipedia tools"""

        # ------- RETRIEVER TOOL -------
        def retriever_tool_fn(query):
            docs = self.retriever.invoke(query)
            if not docs:
                return "No documents found."

            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = getattr(d, "metadata", {}) or {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")

            return "\n\n".join(merged)

        retriever_tool_fn.__annotations__ = {}  # IMPORTANT FIX

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from your vectorstore.",
            func=retriever_tool_fn,
        )

        # ------- WIKIPEDIA TOOL -------
        def wikipedia_tool_fn(query):
            try:
                wiki = WikipediaQueryRun(
                    api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
                )
                return wiki.run(query)
            except Exception as e:
                return f"Wikipedia lookup failed: {e}"

        wikipedia_tool_fn.__annotations__ = {}  # IMPORTANT FIX

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general information.",
            func=wikipedia_tool_fn,
        )

        return [retriever_tool, wikipedia_tool]

    # ----------------------------------------------------------
    # CREATE THE REACT AGENT  (FIXED FOR NEW API)
    # ----------------------------------------------------------
    def _build_agent(self):
        tools = self._build_tools()

        system_prompt = (
            "You are a reasoning agent that uses tools when needed. "
            "Prefer 'retriever' for document-based answers; use 'wikipedia' "
            "for general world knowledge. Always return the final answer only."
        )

        # FIXED — new required signature:
        # create_react_agent(model, tools=[...], prompt="...")
        self._agent = create_react_agent(
            self.llm,          # model (POSitional argument)
            tools=tools,
            prompt=system_prompt
        )

    # ----------------------------------------------------------
    # REACT ANSWER GENERATION
    # ----------------------------------------------------------
    def generate_answer(self, state: RAGState) -> RAGState:

        if self._agent is None:
            self._build_agent()

        result = self._agent.invoke({
            "messages": [HumanMessage(content=state.question)]
        })

        messages = result.get("messages", [])
        answer: Optional[str] = None

        if messages:
            last_msg = messages[-1]
            answer = getattr(last_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer.",
        )
