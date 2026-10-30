"""
app/services/query_rewriter.py
==============================
PURPOSE: Multi-turn conversational session memory and context-aware query reformulation.

Converts ambiguous or pronoun-dependent follow-up questions
(e.g., "What happens when it expires?") into standalone, self-contained search queries
(e.g., "What happens when the semantic cache Time-To-Live expires?") before cache lookup
and document retrieval.
"""

import time
import logging
import threading
from typing import List, Dict, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import get_settings

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """You are an expert search query reformulator.
Given the chat history between a user and an AI assistant, and a follow-up question, your task is to rephrase the follow-up question into a single, complete, standalone search query that can be understood without the chat history.

RULES:
1. Preserve the user's core intent exactly.
2. Resolve all pronouns (it, that, this, they, their, its) into the explicit entities mentioned earlier.
3. If the follow-up question is ALREADY completely standalone, return it EXACTLY as-is.
4. Do NOT answer the question. Only return the reformulated query text.
5. Keep it concise, natural, and factual.

Chat History:
{history}

Follow-up Question:
{question}

Standalone Query:"""


class ConversationSessionManager:
    """Thread-safe multi-turn chat session memory."""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self._sessions: Dict[str, List[Tuple[str, str]]] = {}
        self._lock = threading.Lock()

    def get_history(self, session_id: str) -> List[Tuple[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, question: str, answer: str):
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append((question, answer))
            if len(self._sessions[session_id]) > self.max_history:
                self._sessions[session_id].pop(0)

    def clear_session(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)

    def clear_all(self):
        with self._lock:
            self._sessions.clear()


# Global session manager instance
session_manager = ConversationSessionManager()


class QueryRewriter:
    """Reformulates conversational follow-up questions into standalone queries."""

    def __init__(self):
        pass

    def rewrite(
        self,
        question: str,
        history: Optional[List[Tuple[str, str]]] = None,
        mock_mode: bool = False,
    ) -> str:
        """
        Reformulate the query if conversation history exists.

        Args:
            question: Current user question.
            history: List of (question, answer) pairs from prior turns.
            mock_mode: If True, uses deterministic heuristic substitution for testing.

        Returns:
            Resolved standalone query string.
        """
        if not history or len(history) == 0:
            return question.strip()

        settings = get_settings()
        if not settings.query_rewriting_enabled:
            return question.strip()

        # Format history string
        history_lines = []
        for idx, (q, a) in enumerate(history[-3:], start=1):
            history_lines.append(f"User: {q}\nAssistant: {a[:150]}")
        history_str = "\n".join(history_lines)

        # Quick heuristic for offline mock mode or test suites
        if mock_mode or not settings.openai_api_key or settings.openai_api_key.startswith("your-") or settings.openai_api_key == "sk-test":
            last_q, _last_a = history[-1]
            lower_q = question.lower()
            # Simple pronoun substitution heuristic
            if any(p in lower_q for p in [" it", " its", " that", " this"]):
                # Extract subject from last query
                keywords = [w for w in last_q.replace("?", "").split() if len(w) > 3 and w.lower() not in {"what", "where", "when", "which"}]
                subj = " ".join(keywords[:3]) if keywords else "the entity"
                rewritten = question.replace(" it", f" {subj}").replace(" its", f" {subj}'s").replace(" that", f" {subj}")
                logger.info(f"Heuristic query rewrite: '{question}' -> '{rewritten}'")
                return rewritten
            return question.strip()

        try:
            llm = ChatOpenAI(
                model=settings.llm_model_simple,  # use cheap model for rewriting
                openai_api_key=settings.openai_api_key,
                temperature=0.0,
            )
            prompt = ChatPromptTemplate.from_template(REWRITE_PROMPT)
            chain = prompt | llm | StrOutputParser()

            rewritten = chain.invoke({
                "history": history_str,
                "question": question,
            }).strip()

            logger.info(f"LLM rewritten query: '{question}' -> '{rewritten}'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Falling back to original question.")
            return question.strip()


# Global rewriter instance
query_rewriter = QueryRewriter()
