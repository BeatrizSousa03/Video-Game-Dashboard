import streamlit as st
import pandas as pd
import plotly.express as px
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Game Industry Dashboard", layout="wide")

@st.cache_data
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Data_cleaned.csv')
    
    df = pd.read_csv(file_path)
    
    # Cleaning
    df['User_Score'] = pd.to_numeric(df['User_Score'], errors='coerce')
    df['Year_of_Release'] = pd.to_numeric(df['Year_of_Release'], errors='coerce')
    df.dropna(subset=['Name', 'Genre', 'Year_of_Release', 'Critic_Score', 'User_Score', 'Rating'], inplace=True)
    df['Year_of_Release'] = df['Year_of_Release'].astype(int)
    
    # Maker Logic
    def get_maker(platform):
        if platform in ['PS', 'PS2', 'PS3', 'PS4', 'PSP', 'PSV']: return 'Sony'
        elif platform in ['NES', 'SNES', 'N64', 'GC', 'Wii', 'WiiU', 'Switch', 'GB', 'GBA', 'DS', '3DS']: return 'Nintendo'
        elif platform in ['XB', 'X360', 'XOne']: return 'Microsoft'
        elif platform == 'PC': return 'PC'
        else: return 'Other'
        
    df['Maker'] = df['Platform'].apply(get_maker)
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("Filter Data")

# A. REGION FILTER (Added Back!)
region_options = {
    "Global": "Global_Sales",
    "North America": "NA_Sales",
    "Europe": "EU_Sales",
    "Japan": "JP_Sales",
    "Other": "Other_Sales"
}
selected_region_label = st.sidebar.selectbox("🌍 Focus Region (Sales Only)", list(region_options.keys()))
selected_sales_col = region_options[selected_region_label]

st.sidebar.markdown("---")

# B. STANDARD FILTERS
min_limit = 1994
max_limit = 2016

selected_years = st.sidebar.slider(
    "Select Year Range", 
    min_value=min_limit, 
    max_value=max_limit, 
    value=(min_limit, max_limit)
)

all_genres = sorted(df['Genre'].unique())
selected_genres = st.sidebar.multiselect("Select Genres", all_genres, default=all_genres)

# --- APPLYING FILTERS (SCOPED LOGIC) ---

# 1. GLOBAL SCOPE (For Heatmap, Ratings, Score KPIs)
df_global_scope = df[
    (df['Year_of_Release'].between(selected_years[0], selected_years[1])) &
    (df['Genre'].isin(selected_genres))
]

# 2. REGIONAL SCOPE (For Sales Charts)
# Filters out games that had 0 sales in the selected region
df_regional_scope = df_global_scope[df_global_scope[selected_sales_col] > 0]

if df_global_scope.empty:
    st.warning("No data matches your filters! Please adjust the sliders.")
    st.stop()

# --- 3. MAIN DASHBOARD ---
st.title("🎮 Video Game Industry Analysis")
st.markdown(f"**Focus Region:** {selected_region_label}")

# ==========================================
# ROW 1: KEY METRICS (Mixed Scope)
# ==========================================
col1, col2, col3, col4 = st.columns(4)
# Sales metrics use the REGIONAL scope
col1.metric(f"Total Sales ({selected_region_label})", f"{df_regional_scope[selected_sales_col].sum():,.1f}M")
col2.metric("Games Analyzed", f"{len(df_regional_scope)}")
# Quality metrics use the GLOBAL scope (Stable)
col3.metric("Average Critic Score", f"{df_global_scope['Critic_Score'].mean():.1f}")
col4.metric("Average User Score", f"{df_global_scope['User_Score'].mean():.1f}")

st.markdown("---")

# ==========================================
# ROW 2: RATINGS DISTRIBUTION (Global Scope)
# ==========================================
# 1. Calculate Counts
rating_counts = df_global_scope['Rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Count']

# 2. Strict Filter: Drop 0 values
rating_counts = rating_counts[rating_counts['Count'] > 0]

# 3. Create Donut Chart
fig_rating = px.pie(
    rating_counts, 
    names='Rating', 
    values='Count', 
    hole=0.4, 
    title="<b>Distribution of Games by ESRB Rating (Global)</b>",
    color_discrete_sequence=px.colors.sequential.RdBu
)

# --- UPDATE: Labels outside, Legend removed ---
fig_rating.update_traces(textposition='outside', textinfo='label+percent')
fig_rating.update_layout(
    showlegend=False, 
    height=300,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig_rating, use_container_width=True)


# ==========================================
# ROW 3: CONSOLE WARS (Regional Scope + Colorblind Safe)
# ==========================================
st.subheader(f"1. The Console Wars in {selected_region_label} (Market Evolution)")

# Update Groupby to use selected_sales_col
df_trend = df_regional_scope.groupby(['Year_of_Release', 'Maker'])[selected_sales_col].sum().reset_index()

# --- COLORBLIND FRIENDLY PALETTE (Okabe-Ito Inspired) ---
cb_palette = {
    'Sony': '#0072B2',      # Strong Blue
    'Nintendo': '#D55E00',  # Vermilion (Red/Orange)
    'Microsoft': '#009E73', # Bluish Green (Teal)
    'PC': '#333333',        # Dark Grey
    'Other': '#999999'      # Light Grey
}

fig_trend = px.line(
    df_trend, 
    x='Year_of_Release', 
    y=selected_sales_col, # Dynamic Y-Axis
    color='Maker',
    title=f'Sales Volume by Manufacturer ({selected_region_label})',
    color_discrete_map=cb_palette, # <-- APPLIED HERE
    template="plotly_white"
)
fig_trend.update_layout(height=400, hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)


# ==========================================
# ROW 4: STRATEGY & CONSENSUS
# ==========================================
row4_col1, row4_col2 = st.columns(2)

with row4_col1:
    st.subheader(f"2. Strategy Matrix ({selected_region_label})")
    
    # Aggregating sales based on the selected region
    df_genre = df_regional_scope.groupby('Genre').agg({
        selected_sales_col: 'mean',  # Dynamic Sales
        'Critic_Score': 'mean',
        'Name': 'count'
    }).reset_index()
    
    fig_matrix = px.scatter(
        df_genre,
        x='Critic_Score',
        y=selected_sales_col, # Dynamic Y-Axis
        size='Name',
        color='Genre',
        text='Genre',
        size_max=60,
        title=f'Strategy Matrix: Quality vs. Reach ({selected_region_label})',
        labels={selected_sales_col: 'Avg Sales (M)', 'Critic_Score': 'Avg Critic Score'},
        template="plotly_white"
    )
    
    mid_x = df_genre['Critic_Score'].median() if not df_genre.empty else 70
    mid_y = df_genre[selected_sales_col].median() if not df_genre.empty else 0.5
    fig_matrix.add_hline(y=mid_y, line_dash="dot", line_color="grey")
    fig_matrix.add_vline(x=mid_x, line_dash="dot", line_color="grey")
    
    fig_matrix.update_traces(textposition='top center', textfont_size=11, textfont_weight='bold')
    
    st.plotly_chart(fig_matrix, use_container_width=True)

with row4_col2:
    st.subheader("3. Consensus Heatmap (Global)")
    
    # Using Global Scope for Consensus to keep it stable
    df_grouped = df_global_scope.groupby('Name').agg({
        'Critic_Score': 'mean',
        'User_Score': 'mean'
    }).reset_index()
    df_grouped['User_Score_100'] = df_grouped['User_Score'] * 10
    df_clean = df_grouped.dropna(subset=['Critic_Score', 'User_Score_100'])

    if len(df_clean) > 5:
        # --- WHITE TEXT STYLING FOR DARK MODE ---
        plt.rcParams.update({
            'text.color': 'white',
            'axes.labelcolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'font.family': 'sans-serif'
        })
        
        # Colormap
        cmap_base = plt.cm.Blues
        cmap_lighter_start = mcolors.LinearSegmentedColormap.from_list(
            'trunc_blues', cmap_base(np.linspace(0.2, 1.0, 256))
        )
        cmap_lighter_start.set_under('white')

        # Plotting
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Transparent background
        fig.patch.set_alpha(0) 
        ax.patch.set_alpha(0)

        hb = ax.hexbin(
            df_clean['Critic_Score'],
            df_clean['User_Score_100'],
            gridsize=25,
            cmap=cmap_lighter_start,
            mincnt=1,
            edgecolors='white',
            linewidths=0.5
        )

        # Lines
        ax.plot([0, 100], [0, 100], ls='--', color='white', linewidth=1.5, label='Perfect Agreement', alpha=0.7)

        if len(df_clean) > 1:
            x = df_clean['Critic_Score']
            y = df_clean['User_Score_100']
            m, b = np.polyfit(x, y, 1)
            # Trendline: Neon Red
            ax.plot(x, m * x + b, color='#FF0055', linewidth=3, label='Trendline (Actual)')

        ax.set_title("User vs. Critic Agreement Density", fontsize=12, fontweight='bold', pad=10, loc='left')
        ax.set_xlabel("Critic Score", fontsize=10)
        ax.set_ylabel("User Score (Scaled)", fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        
        # Clean Spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')
        
        # Legend (Black Text on White Box)
        leg = ax.legend(loc='upper left', frameon=True, fontsize=9, facecolor='white', framealpha=0.9, edgecolor='white')
        for text in leg.get_texts():
            text.set_color("black")
        
        st.pyplot(fig)
    else:
        st.warning("Not enough data points.")