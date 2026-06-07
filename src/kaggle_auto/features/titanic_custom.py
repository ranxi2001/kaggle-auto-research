"""Custom Titanic features: domain-specific feature engineering."""

import re

import numpy as np
import pandas as pd

from .base import BaseFeature
from .registry import register


@register
class TitanicFeatures(BaseFeature):
    """Domain-specific features for Titanic competition."""

    name = "titanic_custom"
    dependencies = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        # Sex
        if "Sex" in df.columns:
            result["Sex_encoded"] = (df["Sex"] == "male").astype(int)

        # Title
        if "Name" in df.columns:
            result["Title_encoded"] = (
                df["Name"]
                .apply(lambda x: re.search(r" ([A-Za-z]+)\.", str(x)))
                .apply(lambda x: x.group(1) if x else "Unknown")
                .map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Dr": 4, "Rev": 5})
                .fillna(6)
            )

        # Cabin
        if "Cabin" in df.columns:
            deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8, "U": 0}
            result["Deck_encoded"] = (
                df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U").map(deck_map).fillna(0)
            )
            result["HasCabin"] = df["Cabin"].notna().astype(int)

        # Family
        if "SibSp" in df.columns and "Parch" in df.columns:
            family = df["SibSp"] + df["Parch"] + 1
            result["FamilySize"] = family
            result["IsAlone"] = (family == 1).astype(int)

        # Fare
        if "Fare" in df.columns:
            fare = df["Fare"].fillna(df["Fare"].median())
            family_size = result.get("FamilySize", pd.Series(1, index=df.index))
            result["FarePerPerson"] = fare / family_size.clip(lower=1)
            result["FareBin"] = pd.qcut(fare, q=5, labels=False, duplicates="drop")

        # Age
        if "Age" in df.columns:
            age = df["Age"].fillna(df["Age"].median())
            result["AgeFilled"] = age
            result["AgeBin"] = pd.cut(age, bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)
            result["IsChild"] = (age < 12).astype(int)

        # Embarked
        if "Embarked" in df.columns:
            result["Embarked_encoded"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)

        # Interactions
        if "Sex_encoded" in result.columns and "Pclass" in df.columns:
            result["Sex_Pclass"] = result["Sex_encoded"] * 10 + df["Pclass"]

        return result
