"""Text/NLP feature generators for LLM competitions."""

import numpy as np
import pandas as pd

from .base import BaseFeature
from .registry import register


@register
class TextStats(BaseFeature):
    """Basic text statistics."""

    name = "text_stats"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        text_cols = df.select_dtypes(include=["object"]).columns
        result = pd.DataFrame(index=df.index)

        for col in text_cols:
            series = df[col].fillna("")
            if series.str.len().mean() < 20:
                continue

            result[f"{col}_len"] = series.str.len()
            result[f"{col}_word_count"] = series.str.split().str.len()
            result[f"{col}_sentence_count"] = series.str.count(r"[.!?]+")
            result[f"{col}_avg_word_len"] = (
                series.str.len() / series.str.split().str.len().replace(0, 1)
            )
            result[f"{col}_uppercase_ratio"] = (
                series.str.count(r"[A-Z]") / series.str.len().replace(0, 1)
            )
            result[f"{col}_digit_ratio"] = (
                series.str.count(r"\d") / series.str.len().replace(0, 1)
            )
            result[f"{col}_special_char_ratio"] = (
                series.str.count(r"[^a-zA-Z0-9\s]") / series.str.len().replace(0, 1)
            )

        return result


@register
class TokenCount(BaseFeature):
    """Approximate token count (whitespace-based, no tokenizer dependency)."""

    name = "token_count"

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        text_cols = df.select_dtypes(include=["object"]).columns
        result = pd.DataFrame(index=df.index)

        for col in text_cols:
            series = df[col].fillna("")
            if series.str.len().mean() < 20:
                continue
            result[f"{col}_approx_tokens"] = series.str.split().str.len() * 1.3

        return result
