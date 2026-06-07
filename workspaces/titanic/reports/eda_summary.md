# EDA Summary Report

## Dataset Overview

- **Rows**: 891
- **Columns**: 12
- **Memory**: 0.3 MB
- **Duplicate rows**: 0

- **Numeric columns**: 7
- **Categorical columns**: 5
- **Datetime columns**: 0

## Target Variable

- **Name**: `Survived`
- **Type**: classification
- **Classes**: 2
- **Imbalance ratio**: 1.61

## Data Quality

**High null columns** (>30%): `Cabin`

**High cardinality** (>95%): `PassengerId`, `Name`

## Column Details

| Column | Type | Null% | Unique | Notes |
|--------|------|-------|--------|-------|
| `PassengerId` | int64 | 0.0% | 891 |  |
| `Survived` | int64 | 0.0% | 2 |  |
| `Pclass` | int64 | 0.0% | 3 |  |
| `Name` | object | 0.0% | 891 |  |
| `Sex` | object | 0.0% | 2 |  |
| `Age` | float64 | 19.9% | 88 |  |
| `SibSp` | int64 | 0.0% | 7 | skew=3.7 |
| `Parch` | int64 | 0.0% | 7 | skew=2.7 |
| `Ticket` | object | 0.0% | 681 |  |
| `Fare` | float64 | 0.0% | 248 | skew=4.8 |
| `Cabin` | object | 77.1% | 147 | high null |
| `Embarked` | object | 0.2% | 3 |  |

## Train/Test Distribution Shift

| Column | Train Mean | Test Mean | Shift Score |
|--------|-----------|----------|-------------|
| `PassengerId` | 446.0000 | 1100.5000 | 2.54 |
