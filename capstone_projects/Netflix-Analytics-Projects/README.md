# 🎬 Netflix Business Analytics and Visualization Project

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📋 Project Overview

This project presents a **comprehensive business analytics study** of Netflix's content library using Python. Through rigorous data cleaning, exploratory data analysis (EDA), and professional-grade visualizations, we uncover actionable insights about Netflix's content strategy, global distribution, and growth trends.

The analysis examines **8,800+ titles** spanning Movies and TV Shows from multiple countries, genres, and rating categories to answer critical business questions about Netflix's content landscape.

### 🎯 Key Objectives

- Analyze the composition and distribution of Netflix's content catalog
- Identify global content production trends and top contributing countries
- Explore temporal patterns in content additions and releases
- Examine rating distributions and audience targeting strategies
- Uncover duration patterns across different content types
- Generate actionable business insights for strategic decision-making

---

## 📊 Dataset Description

| Feature | Description |
|---|---|
| **Source** | Netflix Titles Dataset |
| **Records** | 8,807 titles |
| **Features** | 12 columns |
| **Content Types** | Movies & TV Shows |
| **Time Span** | 1925–2021 |

### Dataset Columns

| Column | Description |
|---|---|
| `show_id` | Unique identifier for each title |
| `type` | Movie or TV Show |
| `title` | Name of the content |
| `director` | Director(s) of the title |
| `cast` | Lead actors/actresses |
| `country` | Country of production |
| `date_added` | Date added to Netflix |
| `release_year` | Year of original release |
| `rating` | Content rating (e.g., PG-13, TV-MA) |
| `duration` | Duration in minutes (Movies) or seasons (TV Shows) |
| `listed_in` | Genre categories |
| `description` | Brief synopsis |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computing |
| **Matplotlib** | Static visualizations and chart customization |
| **Seaborn** | Statistical data visualization |
| **Jupyter Notebook** | Interactive development environment |

---

## 📁 Folder Structure

```
Netflix-Analytics-Projects/
│
├── data/
│   └── netflix_titles.csv          # Raw dataset
│
├── notebooks/
│   └── netflix_analysis.ipynb      # Main analysis notebook
│
├── images/                          # Saved visualization exports
│
├── README.md                        # Project documentation (this file)
├── requirements.txt                 # Python dependencies
└── .gitignore                       # Git ignore rules
```

---

## 🔍 Key Analyses Performed

### 1. Data Cleaning & Preprocessing
- Handled missing values across 6 columns with strategic imputation
- Converted date strings to proper datetime format
- Engineered new features: `year_added`, `month_added`, `duration_int`
- Separated movie durations (minutes) from TV show durations (seasons)

### 2. Content Type Analysis
- **Movies dominate** Netflix's library at ~69.6% vs TV Shows at ~30.4%
- Countplot and pie chart analysis of content distribution

### 3. Global Content Production
- **United States** leads content production with 2,800+ titles
- **India** ranks second with 970+ titles
- Top 10 country analysis with annotated bar charts

### 4. Temporal Trends
- Exponential growth in content additions from 2015 onward
- Peak content addition year identified as **2019**
- Monthly and yearly trend analysis with line plots

### 5. Rating Distribution
- **TV-MA** is the most frequent rating, indicating a focus on mature audiences
- Cross-tabulation of ratings by content type (Movies vs TV Shows)

### 6. Duration Analysis
- Average movie duration: ~100 minutes
- Distribution analysis using histograms, KDE plots, boxplots, and violin plots
- Jointplot analysis of release year vs duration relationship

### 7. Genre & Category Insights
- Most popular genres identified through listed_in analysis
- International content represents a significant and growing portion

---

## 📈 Sample Visualizations

The project includes **12+ professional visualizations**:

| # | Visualization | Insight |
|---|---|---|
| 1 | **Countplot** | Movies vs TV Shows distribution |
| 2 | **Histogram** | Release year distribution patterns |
| 3 | **Boxplot** | Movie duration outlier analysis |
| 4 | **Heatmap** | Numerical feature correlations |
| 5 | **Barplot** | Top 10 content-producing countries |
| 6 | **Violin Plot** | Release year distribution by type |
| 7 | **Pairplot** | Multi-feature relationship matrix |
| 8 | **KDE Plot** | Movie duration density curve |
| 9 | **Jointplot** | Release year × duration relationship |
| 10 | **Pie Chart** | Content type percentage breakdown |
| 11 | **Lineplot** | Content growth over time |
| 12 | **Countplot (hue)** | Rating breakdown by content type |

---

## 💡 Business Insights

1. **Content Strategy**: Netflix's catalog is heavily skewed toward Movies (~70%), suggesting a strategic emphasis on film content for broader audience appeal.

2. **Global Expansion**: While the US dominates production, India's strong second-place position reflects Netflix's aggressive expansion into the Indian market.

3. **Growth Trajectory**: Content additions showed exponential growth from 2015–2019, with a slight decline in 2020–2021 likely due to pandemic-related production delays.

4. **Audience Targeting**: The prevalence of TV-MA ratings indicates Netflix primarily targets adult audiences, with family-friendly content representing a smaller but significant segment.

5. **Duration Optimization**: Most movies cluster around 90–120 minutes, aligning with industry-standard feature film lengths and audience attention spans.

6. **International Focus**: A substantial portion of content originates from non-English-speaking countries, reflecting Netflix's commitment to diverse, global storytelling.

---

## 🚀 Future Improvements

- [ ] Implement **NLP-based sentiment analysis** on content descriptions
- [ ] Build a **content recommendation engine** using collaborative filtering
- [ ] Create an **interactive dashboard** with Plotly/Dash or Streamlit
- [ ] Perform **time-series forecasting** for future content growth predictions
- [ ] Add **genre-level deep dive** analysis with network graphs
- [ ] Integrate **IMDb ratings** for cross-platform quality analysis

---

## ⚙️ How to Run the Project

### Prerequisites
- Python 3.9 or higher
- Jupyter Notebook

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Netflix-Analytics-Projects.git
   cd Netflix-Analytics-Projects
   ```

2. **Activate the virtual environment**
   ```bash
   # Windows
   mlenv\Scripts\activate

   # macOS/Linux
   source mlenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook notebooks/netflix_analysis.ipynb
   ```

5. **Run all cells** to reproduce the analysis and visualizations.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Netflix-Analytics-Projects/issues).

---

<p align="center">
  <b>⭐ If you found this project useful, please consider giving it a star!</b>
</p>
