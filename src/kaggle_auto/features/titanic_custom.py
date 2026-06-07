"""Custom Titanic features: advanced domain-specific feature engineering."""

import re

import numpy as np
import pandas as pd

from .base import BaseFeature
from .registry import register


@register
class TitanicFeatures(BaseFeature):
    """Advanced domain-specific features for Titanic competition."""

    name = "titanic_custom"
    dependencies = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        # Title extraction with grouping
        if "Name" in df.columns:
            titles = df["Name"].apply(
                lambda x: re.search(r" ([A-Za-z]+)\.", str(x)).group(1)
                if re.search(r" ([A-Za-z]+)\.", str(x)) else "Unknown"
            )
            title_group = {
                "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
                "Dr": "Officer", "Rev": "Officer", "Col": "Officer",
                "Major": "Officer", "Capt": "Officer",
                "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
                "Lady": "Royalty", "Countess": "Royalty", "Sir": "Royalty",
                "Don": "Royalty", "Dona": "Royalty", "Jonkheer": "Royalty",
            }
            grouped = titles.map(title_group).fillna("Other")
            title_enc = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Officer": 4, "Royalty": 5, "Other": 6}
            result["TitleGroup_enc"] = grouped.map(title_enc).fillna(6)
        else:
            grouped = None

        # Sex
        if "Sex" in df.columns:
            result["Sex_enc"] = (df["Sex"] == "male").astype(int)

        # Age with title-group imputation
        if "Age" in df.columns:
            age = df["Age"].copy()
            if grouped is not None:
                age_medians = df.assign(_grp=grouped).groupby("_grp")["Age"].transform("median")
                age = age.fillna(age_medians)
            age = age.fillna(age.median())
            result["AgeFilled"] = age
            result["AgeBin"] = pd.cut(age, bins=[0, 5, 12, 18, 25, 35, 50, 65, 100], labels=False).fillna(3).astype(float)
            result["IsChild"] = (age <= 12).astype(int)
            result["IsElderly"] = (age >= 60).astype(int)

        # Family
        if "SibSp" in df.columns and "Parch" in df.columns:
            family = df["SibSp"] + df["Parch"] + 1
            result["FamilySize"] = family
            result["IsAlone"] = (family == 1).astype(int)
            result["SmallFamily"] = ((family >= 2) & (family <= 4)).astype(int)
            result["LargeFamily"] = (family >= 5).astype(int)

        # Ticket group size
        if "Ticket" in df.columns:
            ticket_counts = df["Ticket"].value_counts()
            result["TicketGroupSize"] = df["Ticket"].map(ticket_counts).fillna(1)
            result["SharedTicket"] = (result["TicketGroupSize"] > 1).astype(int)

        # Cabin
        if "Cabin" in df.columns:
            result["HasCabin"] = df["Cabin"].notna().astype(int)
            deck = df["Cabin"].apply(lambda x: str(x)[0] if pd.notna(x) else "U")
            deck_surv = {"A": 0.47, "B": 0.74, "C": 0.59, "D": 0.76, "E": 0.75, "F": 0.62, "G": 0.50, "T": 0.0, "U": 0.30}
            result["Deck_surv_rate"] = deck.map(deck_surv).fillna(0.3)

        # Fare
        if "Fare" in df.columns:
            fare = df["Fare"].fillna(df["Fare"].median())
            family_size = result.get("FamilySize", pd.Series(1, index=df.index))
            result["FarePerPerson"] = fare / family_size.clip(lower=1)
            result["FareBin"] = pd.qcut(fare, q=6, labels=False, duplicates="drop")
            result["HighFare"] = (fare > fare.quantile(0.75)).astype(int)

        # Embarked
        if "Embarked" in df.columns:
            result["Embarked_enc"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(0)

        # Interactions
        pclass = df["Pclass"] if "Pclass" in df.columns else pd.Series(2, index=df.index)
        if "Sex_enc" in result.columns:
            result["Sex_Pclass"] = result["Sex_enc"] * 3 + pclass
        if "AgeBin" in result.columns:
            result["Age_Pclass"] = result["AgeBin"] * 3 + pclass
        if "FamilySize" in result.columns:
            result["FamilySize_Pclass"] = result["FamilySize"] * pclass
        if "IsChild" in result.columns and "Sex_enc" in result.columns:
            result["Child_or_Female"] = ((result["IsChild"] == 1) | (result["Sex_enc"] == 0)).astype(int)

        return result
