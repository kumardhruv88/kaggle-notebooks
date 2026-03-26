# 🛒 Customer Segmentation with K-Means Clustering
### RFM Analysis on UK E-Commerce Retail Data (1M+ Transactions)

---

## 📌 Project Overview

This project performs **end-to-end customer segmentation** on a real-world UK-based e-commerce dataset containing over **1 million transactions** spanning 2 years (Dec 2009 – Dec 2011). Using **RFM (Recency, Frequency, Monetary) feature engineering** and **K-Means clustering**, customers are grouped into 4 actionable business segments — enabling targeted marketing, churn prevention, and revenue optimization strategies.

> **Core Question:** *Given raw transactional data with no pre-built features, can we automatically identify distinct customer personas that a business can act on?*

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Online Retail II UCI — Kaggle |
| **Raw Size** | 1,067,371 rows × 8 columns |
| **Time Period** | December 2009 – December 2011 |
| **Geography** | 43 countries |
| **Unique Customers (raw)** | 5,942 |
| **Unique Invoices** | 53,628 |

**Columns:** `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`

---

## 🔬 Methodology & Pipeline

### Step 1 — Exploratory Data Analysis
Performed comprehensive EDA to understand data quality before any modelling decisions:

- **22.77% of rows (243,007) had missing Customer IDs** — identified as the largest data quality issue
- **19,494 cancelled transactions** (invoices prefixed with `"C"`) found, representing 1.83% of data
- **22,950 rows with negative Quantity** identified as returns/reversals
- **5 rows with negative Price** and **6,202 rows with zero Price** flagged as erroneous entries
- Price ranged from **–53,594 to +38,970** — extreme outliers confirmed bad data entries
- Quantity ranged from **–80,995 to +80,995** — symmetric extremes suggesting data entry artifacts

---

### Step 2 — Data Cleaning
Applied a **7-step cleaning pipeline** in intentional order (order matters — type conversion after null removal to avoid crashes):

| Step | Action | Rows Removed | Reason |
|---|---|---|---|
| 1 | Drop null Customer IDs | 243,007 | Cannot attribute transactions to any customer |
| 2 | Remove cancelled invoices (prefix "C") | 18,744 | Returns skew monetary value negatively |
| 3 | Remove negative/zero Quantity | 0 additional | Already covered by step 2 |
| 4 | Remove negative/zero Price | 71 | Erroneous pricing entries |
| 5 | Cast Customer ID → `int64` | — | Enables correct groupby aggregation |
| 6 | Cast InvoiceDate → `datetime64` | — | Required for recency calculation |
| 7 | Engineer `TotalPrice = Quantity × Price` | — | Base metric for monetary feature |

**Final clean dataset: 805,549 rows × 9 columns | 5,878 unique customers**

---

### Step 3 — RFM Feature Engineering
Transformed transaction-level data into a **customer-level feature matrix** — the core engineering challenge of this project.

> **Why RFM?** It captures the three most predictive dimensions of customer value from raw purchase history alone, without any demographic data.

| Feature | Definition | Calculation |
|---|---|---|
| **Recency (R)** | Days since last purchase | `(reference_date - last_invoice_date).days` |
| **Frequency (F)** | Number of unique orders placed | `COUNT(DISTINCT Invoice)` per customer |
| **Monetary (M)** | Total revenue generated | `SUM(TotalPrice)` per customer |

**Reference date:** December 10, 2011 (1 day after last transaction — simulates "today")

**RFM Summary Statistics:**

| Metric | Recency (days) | Frequency (orders) | Monetary (£) |
|---|---|---|---|
| Mean | 201 | 6.3 | 3,019 |
| Median | 96 | 3 | 899 |
| Max | 739 | 398 | 608,822 |
| Std Dev | 209 | 13 | 14,738 |

---

### Step 4 — Statistical Preprocessing

**Problem:** K-Means uses Euclidean distance. With raw features, Monetary (range: £3 – £608,822) would completely dominate Recency (range: 1–739 days) and Frequency (range: 1–398), making those dimensions invisible.

**Two-stage fix applied:**

**Stage 1 — Log Transformation** (to fix skewness):

| Feature | Skewness Before | Skewness After | Verdict |
|---|---|---|---|
| Recency | 0.887 | –0.489 | ✅ Normal range |
| Frequency | 12.640 | 1.001 | ✅ Greatly improved |
| Monetary | **25.314** | **0.265** | ✅ Near perfect bell curve |

Used `np.log1p(x)` (= log(1+x)) to safely handle potential zero values.

**Stage 2 — StandardScaler** (to equalize feature scales):

Applied *after* log transform — because StandardScaler is sensitive to outliers. Log-transforming first brings extreme values closer to the distribution before scaling.

**Post-scaling result:** All three features achieved **mean ≈ 0.000** and **std = 1.000** exactly.

---

### Step 5 — Model Selection (Finding Optimal K)

Evaluated K = 2 through 10 using two complementary metrics:

| K | Inertia (WCSS) | Silhouette Score |
|---|---|---|
| 2 | 8,588.81 | 0.4343 |
| 3 | 6,351.98 | 0.3458 |
| **4** | **4,918.64** | **0.3649** ← local peak |
| 5 | 4,097.83 | 0.3462 |
| 6 | 3,552.98 | 0.3354 |
| 10 | 2,455.91 | 0.2856 |

**Chosen K = 4** based on:
- Elbow in the inertia curve flattens after K=4
- Silhouette score has a **local maximum at K=4** (rises back up from K=3 dip)
- K=2, while mathematically optimal, provides no actionable business granularity
- 4 segments map cleanly to real marketing personas

**Algorithm configuration:** `KMeans(init='k-means++', n_init=10, random_state=42)` — k-means++ initialization used to avoid bad local minima from random centroid placement.

---

### Step 6 — Cluster Profiling & Business Interpretation

| Cluster | Segment Name | Customers | % Share | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|---|
| 0 | 🏆 Champions | 1,188 | 20.2% | 27 days | 19 orders | £11,014 |
| 1 | 💀 Lost Customers | 1,974 | 33.6% | 396 days | 1.4 orders | £326 |
| 2 | ⚠️ At Risk | 1,465 | 24.9% | 228 days | 5 orders | £2,002 |
| 3 | 🌱 Promising | 1,251 | 21.3% | 28 days | 3 orders | £865 |

**Segment Interpretation:**
- **Champions** — Bought recently, buy frequently, spend the most. Top 20% of customers likely driving the majority of revenue
- **Lost Customers** — Haven't purchased in over a year, barely 1–2 lifetime orders. Churned; require aggressive win-back campaigns or archival
- **At Risk** — Were moderately active customers now drifting. Mid-range on all metrics; prime candidates for re-engagement campaigns
- **Promising** — Recently acquired customers with low frequency and spend. Need nurturing and onboarding to convert into Champions

---

### Step 7 — Validation via PCA

Applied **PCA (2 components)** to validate cluster separability:

- **PC1 explained variance: 76.4%**
- **PC2 explained variance: 18.8%**
- **Total: 95.1%** — the 2D projection preserves 95.1% of the original 3D RFM structure

The scatter plot confirms 4 clearly separated spatial regions with natural boundary overlap — expected and healthy in real-world data.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data Manipulation | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Machine Learning | scikit-learn (KMeans, StandardScaler, PCA, silhouette_score) |
| Environment | Kaggle Notebooks (CPU) |

---

## 📁 Repository Structure

```
├── k-means-clustering.ipynb          # Main Kaggle notebook (10 cells)
├── README.md               # This file
```

---

## 💡 Key Takeaways & Interview Insights

1. **Feature engineering > model complexity** — The entire value of this project comes from constructing RFM from raw transactions, not from the clustering algorithm itself
2. **Preprocessing order matters** — Log transform before StandardScaler; null removal before type conversion
3. **Use two metrics for K selection** — Elbow method alone is subjective; Silhouette score provides mathematical validation
4. **Business interpretability is the end goal** — A cluster label (0, 1, 2, 3) is worthless without translating it into a named segment with an action plan
5. **PCA for validation, not just visualization** — 95.1% explained variance confirms the clusters are geometrically meaningful

---

## 📈 Business Impact (Hypothetical Strategy)

| Segment | Recommended Action |
|---|---|
| 🏆 Champions | Loyalty rewards, early product access, referral programs |
| 💀 Lost | One-time heavy discount win-back email; archive if no response |
| ⚠️ At Risk | Personalized re-engagement campaign, time-limited offers |
| 🌱 Promising | Welcome series, small incentives, education on product range |

---


