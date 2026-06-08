# DRW Crypto Market Prediction - Research Notes

## Competition Summary

- **Sponsor**: DRW (top quantitative trading firm)
- **Prize**: $25,000
- **Teams**: 1,091
- **Metric**: R-squared (R2)
- **Type**: Regression (predict crypto market returns)
- **Deadline**: 2025-07-25

## Data Schema

- **Train**: 525,887 rows x 897 columns (3.3 GB parquet)
- **Test**: 538,150 rows x 896 columns (3.4 GB parquet)
- **Features**: 890 anonymized proprietary market features (X0-X889) + 5 basic features
- **Target**: Continuous value (crypto return prediction)
- **No missing values** (confirmed by EDA notebook)
- **27 features with single unique value** (constant, should drop)
- **21 feature pairs with perfect correlation** (redundant, keep one)

## Key Insights from EDA

1. **Anonymized features**: All 890 features are named X0-X889, no domain meaning visible
2. **Low signal-to-noise**: Competition title "Low Signal to Noise", very noisy data
3. **Memory optimization**: Raw 3.6GB to 1GB after downcasting (73% reduction needed)
4. **Feature preprocessing needed**: Drop constant columns, deduplicate perfect correlations

## Top Notebook Approaches (by votes)

### 1. Ensemble (449 votes) - ravaghi/drw-crypto-market-prediction-ensemble
- LightGBM + XGBoost ensemble
- Feature selection + denoising
- Likely uses time-based CV split

### 2. XGB+LGBM Ensemble (301 votes) - yehqysns/xgb-lgbm-ensemble
- Straightforward XGB + LightGBM blend
- Strong baseline approach

### 3. DRW Remix VI (181 votes) - bakuer30/drw-remix-vi
- Iterative refinement of public notebooks
- Anti-aliasing / denoising techniques

### 4. Simple H-Blend (179 votes) - nina2025/simple-h-blend
- Blending strategy (horizontal blend = different model types)

### 5. Low Signal to Noise (175 votes) - taylorsamarel/low-signal-to-noise-updated
- Focuses on denoising strategies
- Signal extraction from noisy features

### 6. Anti-Aliasing Technology (154 votes) - nina2025
- Noise reduction in predictions
- Smoothing/filtering techniques

## Recommended Strategy

### Phase 1: Baseline
1. Load data with memory optimization (downcast float64 to float32)
2. Drop 27 constant features
3. Remove one from each perfectly correlated pair
4. LightGBM regression with default params
5. Time-based CV split, not random, because this is time series

### Phase 2: Denoising & Feature Selection
1. Feature importance, keep top 100-200
2. PCA/SVD for dimensionality reduction
3. Rolling statistics if time ordering exists
4. Noise reduction on predictions (post-processing)

### Phase 3: Ensemble
1. LightGBM + XGBoost + CatBoost
2. Different feature subsets per model
3. Weighted blend optimized on CV

### Critical Considerations
- **Time-based split**: This is financial data, future cannot leak into past
- **Low R2**: Expect very low R2 values (0.01-0.05 is normal for financial data)
- **Denoising**: More important than complex features for noisy financial data
- **Anti-overfitting**: Strong regularization needed, few leaves, high min_child
- **Feature selection**: 890 features is too many, most are noise

## IdeaPool Seeds

1. [HIGH] Drop constant + deduplicate correlated features
2. [HIGH] Time-based CV (GroupKFold by time period)
3. [HIGH] LGB + XGB ensemble baseline
4. [MEDIUM] Feature importance to top-K selection (K=100-300)
5. [MEDIUM] PCA/SVD feature reduction as alternative
6. [MEDIUM] Prediction clipping/smoothing (anti-aliasing)
7. [LOW] Target denoising (Kalman filter on residuals)
8. [LOW] Neural network (simple MLP on top features)
