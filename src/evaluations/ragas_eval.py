"""Ragas evaluation integration."""

import os
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness

answer_relevance = AnswerRelevancy()
faithfulness = Faithfulness()

# Ensure OpenAI key is available for Ragas
from src.config import config

os.environ["OPENAI_API_KEY"] = config.openai_api_key


class RagasEvaluator:
    """Evaluates RAG outputs using Ragas library."""

    def __init__(self):
        self.metrics = [faithfulness, answer_relevance]

    def evaluate_response(
        self,
        query: str,
        answer: str,
        retrieved_contexts: list[str],
    ) -> dict[str, float]:
        """
        Run Ragas evaluation on a single response.
        """
        # Ragas expects a dataset
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [retrieved_contexts],
        }
        dataset = Dataset.from_dict(data)

        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            raise_exceptions=False,
        )

        return results.to_pandas().iloc[0].to_dict()
