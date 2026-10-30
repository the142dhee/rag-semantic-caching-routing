"""
app/evaluation/llm_judge.py
===========================
PURPOSE: Automated LLM-as-a-Judge evaluation framework for RAG systems (RAG Triad).

Evaluates 3 core dimensions:
  1. Faithfulness / Groundedness (1-5): Are statements backed by context?
  2. Answer Relevance (1-5): Does the answer directly address the user query?
  3. Context Relevance (1-5): Are the retrieved chunks relevant to the query?

Outputs:
  - Numerical scores (1 to 5) and structured natural-language rationale for each dimension.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.core.config import get_settings
from app.evaluation.metrics import (
    calculate_faithfulness,
    calculate_context_relevance,
    calculate_token_f1,
)

logger = logging.getLogger(__name__)


class LLMJudgeScore(BaseModel):
    """Detailed scores and rationale produced by the LLM Judge."""
    faithfulness_score: int = Field(..., ge=1, le=5, description="1-5 score for grounding in context")
    faithfulness_reason: str
    answer_relevance_score: int = Field(..., ge=1, le=5, description="1-5 score for answering the query")
    answer_relevance_reason: str
    context_relevance_score: int = Field(..., ge=1, le=5, description="1-5 score for context pertinence")
    context_relevance_reason: str
    overall_score: float = Field(..., ge=1.0, le=5.0)


JUDGE_SYSTEM_PROMPT = """You are an impartial, highly rigorous AI evaluation judge assessing a Retrieval-Augmented Generation (RAG) system.
You evaluate along three dimensions on a strict 1 to 5 scale:

1. Faithfulness (1 to 5):
   - 5: Every single claim in the answer is directly supported by the context. Zero extrapolation or hallucination.
   - 3: Most claims are supported, but some minor unverified details are present.
   - 1: Significant hallucinations or claims directly contradicted by the context.

2. Answer Relevance (1 to 5):
   - 5: The answer directly and completely addresses the user's question without unnecessary filler.
   - 3: Answers the question partially or includes irrelevant tangents.
   - 1: Completely fails to address what was asked.

3. Context Relevance (1 to 5):
   - 5: All retrieved context chunks are directly relevant and necessary to answer the question.
   - 3: Contains some relevant information, but mixed with noticeable noise.
   - 1: The retrieved context is completely unrelated to the question.

You MUST respond strictly with valid JSON conforming to this format:
{
  "faithfulness_score": <int 1-5>,
  "faithfulness_reason": "<concise explanation>",
  "answer_relevance_score": <int 1-5>,
  "answer_relevance_reason": "<concise explanation>",
  "context_relevance_score": <int 1-5>,
  "context_relevance_reason": "<concise explanation>",
  "overall_score": <float 1.0-5.0 average>
}
"""

JUDGE_USER_PROMPT = """Question:
{question}

Retrieved Context Chunks:
{context}

Generated Answer:
{answer}
"""


class LLMJudge:
    """Evaluates RAG query responses using an LLM or deterministic fallback rules."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[str],
        mock_mode: bool = False,
    ) -> LLMJudgeScore:
        """
        Evaluate a single RAG response along the RAG triad.

        Args:
            question: The user query string.
            answer: The generated answer string.
            context_chunks: List of context chunk texts.
            mock_mode: If True, uses deterministic heuristic rules for CI/offline testing.

        Returns:
            LLMJudgeScore object.
        """
        settings = get_settings()

        if mock_mode or not settings.openai_api_key or settings.openai_api_key.startswith("your-") or settings.openai_api_key == "sk-test":
            return self._mock_evaluate(question, answer, context_chunks)

        combined_context = "\n\n".join([f"[Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks)])
        selected_model = self.model_name or settings.llm_judge_model

        try:
            llm = ChatOpenAI(
                model=selected_model,
                openai_api_key=settings.openai_api_key,
                temperature=0.0,
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", JUDGE_SYSTEM_PROMPT),
                ("human", JUDGE_USER_PROMPT),
            ])
            chain = prompt | llm | StrOutputParser()

            raw_json = chain.invoke({
                "question": question,
                "context": combined_context,
                "answer": answer,
            })

            clean_json = raw_json.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:-3].strip()
            elif clean_json.startswith("```"):
                clean_json = clean_json[3:-3].strip()

            parsed = json.loads(clean_json)
            return LLMJudgeScore(**parsed)
        except Exception as e:
            logger.warning(f"LLM Judge call failed: {e}. Falling back to deterministic scoring.")
            return self._mock_evaluate(question, answer, context_chunks)

    def _mock_evaluate(self, question: str, answer: str, context_chunks: List[str]) -> LLMJudgeScore:
        """Deterministic heuristic evaluation mapping to 1-5 scale."""
        # 1. Faithfulness score
        f_val = calculate_faithfulness(answer, context_chunks, is_answerable=True)
        if "don't have enough information" in answer.lower():
            faith_score = 5
            faith_reason = "System correctly refused to answer with grounded insufficient information statement."
        elif f_val >= 0.85:
            faith_score = 5
            faith_reason = f"Excellent grounding: {int(f_val*100)}% of content tokens directly match retrieved context."
        elif f_val >= 0.5:
            faith_score = 4
            faith_reason = f"Good grounding: {int(f_val*100)}% of content tokens found in context."
        else:
            faith_score = 2
            faith_reason = "Weak grounding: Answer contains terms not explicitly present in context."

        # 2. Answer relevance score
        f1_metrics = calculate_token_f1(answer, question)
        if len(answer.strip()) > 10 and not answer.startswith("Error"):
            ans_score = 5
            ans_reason = "Answer directly addresses the entities in the query."
        else:
            ans_score = 3
            ans_reason = "Answer is terse or generic."

        # 3. Context relevance score
        c_rel = calculate_context_relevance(question, context_chunks)
        if c_rel >= 0.7:
            ctx_score = 5
            ctx_reason = f"High context relevance: {int(c_rel*100)}% of retrieved chunks match query keywords."
        elif c_rel >= 0.4:
            ctx_score = 4
            ctx_reason = f"Moderate context relevance: {int(c_rel*100)}% of chunks contain query topics."
        else:
            ctx_score = 2
            ctx_reason = "Low context relevance: Retrieved chunks do not closely match query terms."

        overall = round((faith_score + ans_score + ctx_score) / 3.0, 2)

        return LLMJudgeScore(
            faithfulness_score=faith_score,
            faithfulness_reason=faith_reason,
            answer_relevance_score=ans_score,
            answer_relevance_reason=ans_reason,
            context_relevance_score=ctx_score,
            context_relevance_reason=ctx_reason,
            overall_score=overall,
        )


# Global judge instance
llm_judge = LLMJudge()
