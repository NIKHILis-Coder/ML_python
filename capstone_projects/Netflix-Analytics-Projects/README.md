# Netflix Data Analysis Project

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.6+-11557C?style=for-the-badge)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-4C72B0?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## About This Project

I built this project to practice and improve my data analysis skills using a real-world dataset.
The dataset contains information about Movies and TV Shows available on Netflix.

I performed data cleaning, exploratory data analysis (EDA), and created multiple visualizations
to find interesting patterns and trends in Netflix's content library.

---

## Dataset

| Detail | Info |
|---|---|
| **Source** | Netflix Titles Dataset (Kaggle) |
| **Total Records** | 8,807 titles |
| **Columns** | 12 |
| **Content Types** | Movies & TV Shows |
| **Years Covered** | 1925 – 2021 |

### Columns in the Dataset

| Column | What it means |
|---|---|
| `show_id` | Unique ID for each title |
| `type` | Movie or TV Show |
| `title` | Name of the content |
| `director` | Director name |
| `cast` | Main actors |
| `country` | Country where it was made |
| `date_added` | When it was added to Netflix |
| `release_year` | Original release year |
| `rating` | Age rating (e.g., PG-13, TV-MA) |
| `duration` | Length in minutes (Movies) or seasons (TV Shows) |
| `listed_in` | Genre |
| `description` | Short summary |

---

## Tools & Libraries Used

- **Python 3.9+**
- **Pandas** — data cleaning and analysis
- **NumPy** — numerical operations
- **Matplotlib** — creating charts
- **Seaborn** — statistical visualizations
- **Jupyter Notebook** — writing and running code

---

## Project Structure

```
Netflix-Analytics-Projects/
│
├── data/
│   └── netflix_titles.csv        ← dataset file
│
├── notebooks/
│   └── netflix_analysis.ipynb    ← main analysis notebook
│
├── images/                        ← chart screenshots (optional)
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## What I Did in This Project

### Data Cleaning
- Found and filled missing values in 6 columns
- Removed duplicate rows
- Converted `date_added` from text to datetime format
- Created new columns: `year_added`, `month_added`, `duration_int`

### Charts & Visualizations (12 total)

| # | Chart Type | What I Analyzed |
|---|---|---|
| 1 | Countplot | Movies vs TV Shows count |
| 2 | Histogram | Release year distribution |
| 3 | Boxplot | Movie duration spread |
| 4 | Heatmap | Correlation between columns |
| 5 | Barplot | Top 10 countries by content count |
| 6 | Violin Plot | Release year by content type |
| 7 | Pairplot | Relationships between numerical features |
| 8 | KDE Plot | Movie duration density |
| 9 | Jointplot | Release year vs duration |
| 10 | Pie Chart | Movies vs TV Shows percentage |
| 11 | Lineplot | Content added per year |
| 12 | Countplot | Ratings by content type |

---

## Key Findings

1. Netflix has about **70% Movies** and **30% TV Shows**
2. Most content was released after **2010**
3. The average movie on Netflix is around **99 minutes** long
4. **USA** produces the most content, followed by **India**
5. Netflix added the most content in **2019** before slowing down during COVID-19
6. **TV-MA** is the most common rating — Netflix mainly targets adult viewers

---

## Future Ideas

- [ ] Analyze the most popular genres
- [ ] Build a simple movie recommendation system
- [ ] Create an interactive dashboard using Plotly or Streamlit
- [ ] Do sentiment analysis on movie descriptions

---

## How to Run

1. Clone this repository
   ```bash
   git clone https://github.com/NIKHILis-Coder/ML_python.git
   ```

2. Activate the virtual environment
   ```bash
   mlenv\Scripts\activate
   ```

3. Install required libraries
   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook
   ```bash
   jupyter notebook notebooks/netflix_analysis.ipynb
   ```

5. Run all cells from top to bottom.

---
