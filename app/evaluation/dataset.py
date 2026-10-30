"""
app/evaluation/dataset.py
=========================
PURPOSE: Load, validate, and query the structured RAG evaluation dataset.
Supports all 7 evaluation categories:
  1. simple_factual
  2. semantic_variation
  3. complex_question
  4. multi_chunk
  5. unanswerable
  6. routing_test
  7. repeated_question
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "eval_dataset.json"
)


class EvalItem(BaseModel):
    """An individual evaluation test case."""
    id: str
    question: str
    expected_answer: str
    relevant_sources: List[str] = Field(default_factory=list)
    category: str
    is_answerable: bool = True
    expected_tier: str = "SIMPLE"  # SIMPLE or COMPLEX
    variation_of: Optional[str] = None


def load_eval_dataset(path: Optional[str] = None) -> List[EvalItem]:
    """
    Load and parse the evaluation dataset from JSON.

    Args:
        path: Optional file path. Defaults to data/eval_dataset.json.

    Returns:
        List of validated EvalItem objects.
    """
    file_path = path or DEFAULT_DATASET_PATH
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Evaluation dataset not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = [EvalItem(**item) for item in data]
    logger.info(f"Loaded {len(items)} evaluation items from {file_path}")
    return items


def get_items_by_category(category: str, dataset: Optional[List[EvalItem]] = None) -> List[EvalItem]:
    """Filter evaluation items by category."""
    items = dataset or load_eval_dataset()
    return [item for item in items if item.category == category]


def get_dataset_summary(dataset: Optional[List[EvalItem]] = None) -> Dict[str, Any]:
    """Return category distribution and metadata about the dataset."""
    items = dataset or load_eval_dataset()
    categories: Dict[str, int] = {}
    for item in items:
        categories[item.category] = categories.get(item.category, 0) + 1

    return {
        "total_items": len(items),
        "categories": categories,
        "answerable_count": sum(1 for i in items if i.is_answerable),
        "unanswerable_count": sum(1 for i in items if not i.is_answerable),
    }
