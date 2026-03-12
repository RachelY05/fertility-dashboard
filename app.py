import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Adolescent Fertility & Female Education", layout="wide")

# ── 初始化 session state ──────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "EN"
if "edu_choice" not in st.session_state:
    st.session_state.edu_choice = "lower_secondary"
if "year_choice" not in st.session_state:
    st.session_state.year_choice = 2022

# ── 文字内容 ──────────────────────────────────────────
text = {
    "title": {
        "EN": "🌍 Adolescent Fertility Rate & Female Education",
        "ZH": "🌍 青春期生育率与女性受教育程度",
    },
    "subtitle": {
        "EN": "Data source: World Bank Gender Statistics Dataset",
        "ZH": "数据来源：世界银行性别统计数据集",
    },
    "filters": {"EN": "Filters", "ZH": "筛选"},
    "select_year": {"EN": "Select Year", "ZH": "选择年份"},
    "edu_indicator": {"EN": "Education Indicator", "ZH": "教育指标"},
    "lower_secondary": {"EN": "Lower Secondary (%)", "ZH": "初中教育程度 (%)"},
    "upper_secondary": {"EN": "Upper Secondary (%)", "ZH": "高中教育程度 (%)"},
    "scatter_title": {"EN": "Scatter Plot", "ZH": "散点图"},
    "showing": {"EN": "Showing", "ZH": "显示"},
    "countries": {"EN": "countries with available data.", "ZH": "个有数据的国家。"},
    "fertility_label": {
        "EN": "Adolescent Fertility Rate (per 1,000 women)",
        "ZH": "青春期生育率（每千名女性）",
    },
    "map_title": {"EN": "🗺️ World Map", "ZH": "🗺️ 世界地图"},
    "map_subtitle": {
        "EN": "Adolescent fertility rate by country.",
        "ZH": "各国青春期生育率分布。",
    },
    "predictor_title": {"EN": "🔮 Predictor", "ZH": "🔮 预测器"},
    "predictor_subtitle": {
        "EN": "Input a female education rate to predict the adolescent fertility rate.",
        "ZH": "输入女性教育程度，预测对应的青春期生育率。",
    },
    "predicted_label": {
        "EN": "Predicted Adolescent Fertility Rate (per 1,000 women)",
        "ZH": "预测青春期生育率（每千名女性）",
    },
}

def t(key):
    return text[key][st.session_state.lang]

# ── 读取数据 ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("genderstat.csv")
    df["year"] = df["year"].str.extract(r"(\d{4})").astype(int)
    df["adolescent_fertility_rate"] = pd.to_numeric(df["adolescent_fertility_rate"], errors="coerce")
    df["lower_secondary"] = pd.to_numeric(df["lower_secondary"], errors="coerce")
    df["upper_secondary"] = pd.to_numeric(df["upper_secondary"], errors="coerce")
    df["labor_ratio"] = pd.to_numeric(df["labor_ratio"], errors="coerce")
    return df

df = load_data()
years = sorted(df["year"].unique())

# ── 语言切换按钮 ──────────────────────────────────────
col_title, col_lang = st.columns([8, 1])
with col_lang:
    lang = st.session_state.lang
    btn_label = "中文" if lang == "EN" else "English"
    if st.button(btn_label):
        st.session_state.lang = "ZH" if lang == "EN" else "EN"
        st.rerun()

with col_title:
    st.title(t("title"))
st.markdown(t("subtitle"))

# ── 侧边栏（不用 key，完全手动管理）────────────────────
st.sidebar.header(t("filters"))

year_index = years.index(st.session_state.year_choice) if st.session_state.year_choice in years else len(years) - 1
selected_year = st.sidebar.selectbox(t("select_year"), years, index=year_index)
st.session_state.year_choice = selected_year

edu_col_map = {
    "lower_secondary": t("lower_secondary"),
    "upper_secondary": t("upper_secondary"),
}
edu_index = 0 if st.session_state.edu_choice == "lower_secondary" else 1
selected_edu = st.sidebar.radio(
    t("edu_indicator"),
    options=["lower_secondary", "upper_secondary"],
    format_func=lambda x: edu_col_map[x],
    index=edu_index,
)
st.session_state.edu_choice = selected_edu
selected_edu_label = edu_col_map[selected_edu]

# ── 过滤数据 ──────────────────────────────────────────
filtered = df[df["year"] == selected_year].dropna(subset=["adolescent_fertility_rate", selected_edu])

# ── 散点图 ────────────────────────────────────────────
st.subheader(f"{t('scatter_title')} — {selected_year}")
st.write(f"{t('showing')} {len(filtered)} {t('countries')}")

fig = px.scatter(
    filtered,
    x=selected_edu,
    y="adolescent_fertility_rate",
    hover_name="Country Name",
    labels={
        selected_edu: selected_edu_label,
        "adolescent_fertility_rate": t("fertility_label"),
    },
    color="adolescent_fertility_rate",
    color_continuous_scale="RdYlGn_r",
    trendline="ols",
    trendline_color_override="steelblue",
)
st.plotly_chart(fig, use_container_width=True)

# ── 地图 ──────────────────────────────────────────────
st.subheader(t("map_title"))
st.markdown(t("map_subtitle"))

map_data = df[df["year"] == selected_year].dropna(subset=["adolescent_fertility_rate"])
fig_map = px.choropleth(
    map_data,
    locations="Country Code",
    color="adolescent_fertility_rate",
    hover_name="Country Name",
    color_continuous_scale="RdYlGn_r",
    labels={"adolescent_fertility_rate": t("fertility_label")},
)
fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig_map, use_container_width=True)

# ── 预测器 ────────────────────────────────────────────
st.subheader(t("predictor_title"))
st.markdown(t("predictor_subtitle"))

model_data = df.dropna(subset=["adolescent_fertility_rate", selected_edu])
X = model_data[[selected_edu]].values
y = model_data["adolescent_fertility_rate"].values

model = LinearRegression()
model.fit(X, y)

edu_min = float(model_data[selected_edu].min())
edu_max = float(model_data[selected_edu].max())

user_input = st.slider(
    selected_edu_label,
    min_value=edu_min,
    max_value=edu_max,
    value=(edu_min + edu_max) / 2,
)

prediction = model.predict([[user_input]])[0]
st.metric(label=t("predicted_label"), value=f"{prediction:.1f}")