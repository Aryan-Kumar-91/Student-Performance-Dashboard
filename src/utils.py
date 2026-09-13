"""
Module: Utilities & UI Components
Subject: Statistical Machine Learning for Data Science

Provides styling, metric formatters, and academic callouts for a calm,
accessible academic analytics design system.
"""

import streamlit as st

def apply_custom_styles():
    """
    Applies the dashboard's navy, teal, and coral visual system.
    """
    st.markdown("""
        <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
        
        :root {
            /* Palette Tokens */
            --primary: #17324D;
            --primary-deep: #102A43;
            --secondary: #F7FAFC;
            --teal: #0F766E;
            --teal-light: #E6F4F1;
            --sky: #0EA5E9;
            --coral: #D94F4F;
            --accent-light: #EEF6F8;
            --accent-border: #D7E4EA;
            --background-card: #FFFFFF;
            --text-muted: #5B6B7B;
            --border-color: #E4EDF2;
            
            /* Typography Tokens */
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-serif: 'Playfair Display', Georgia, serif;
            
            /* Spacing Tokens */
            --spacing-xs: 0.5rem;
            --spacing-sm: 1.0rem;
            --spacing-md: 2.0rem;
            --spacing-lg: 4.0rem;
            
            /* Radius */
            --radius-sm: 4px;
            --radius-md: 8px;
            --radius-lg: 12px;
        }
        
        /* Global Base */
        html, body, [class*="css"], .stApp {
            font-family: var(--font-sans) !important;
            color: var(--primary) !important;
            background: linear-gradient(135deg, #F7FAFC 0%, #F2F8F8 100%) !important;
            -webkit-font-smoothing: antialiased;
        }
        
        /* Editorial Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font-serif) !important;
            font-weight: 600 !important;
            color: var(--primary) !important;
            letter-spacing: -0.01em;
        }
        
        .main-header {
            font-family: var(--font-serif);
            font-size: 2.7rem;
            font-weight: 600;
            letter-spacing: -0.02em;
            color: var(--primary-deep);
            line-height: 1.15;
            margin-bottom: 0.4rem;
        }
        
        .sub-header {
            font-family: var(--font-sans);
            font-size: 1.05rem;
            color: var(--text-muted);
            margin-bottom: 2rem;
            font-weight: 400;
            line-height: 1.5;
            letter-spacing: 0.01em;
        }
        
        /* Editorial Metric Cards */
        .metric-container {
            background: var(--background-card);
            border-radius: var(--radius-lg);
            padding: 1.35rem 1.25rem;
            border: 1px solid var(--accent-border);
            box-shadow: 0 6px 18px rgba(23, 50, 77, 0.06);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            margin-bottom: 1rem;
        }
        .metric-container:hover {
            transform: translateY(-2px);
            border-color: var(--teal);
            box-shadow: 0 12px 28px rgba(15, 118, 110, 0.12);
        }
        .metric-title {
            font-family: var(--font-sans);
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-muted);
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-family: var(--font-serif);
            font-size: 2.1rem;
            font-weight: 600;
            color: var(--primary);
            line-height: 1.1;
        }
        .metric-subtitle {
            font-family: var(--font-sans);
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-top: 0.45rem;
        }
        
        /* Editorial Educational Card */
        .edu-card {
            background-color: var(--accent-light);
            border-left: 4px solid var(--teal);
            border-radius: var(--radius-sm);
            padding: 1.2rem 1.4rem;
            margin: 1.8rem 0;
            border-top: 1px solid var(--accent-border);
            border-right: 1px solid var(--accent-border);
            border-bottom: 1px solid var(--accent-border);
        }
        .edu-title {
            font-family: var(--font-sans);
            font-weight: 700;
            font-size: 0.92rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--primary);
            margin-bottom: 0.35rem;
        }
        .edu-body {
            font-family: var(--font-sans);
            font-size: 0.92rem;
            color: #29465B;
            line-height: 1.6;
        }
        
        /* Badges & Alerts */
        .alert-safe {
            background-color: #EAF8F5;
            border-left: 3px solid #0F766E;
            border: 1px solid #BFE6DE;
            border-left-width: 4px;
            padding: 1rem 1.2rem;
            border-radius: var(--radius-sm);
            margin: 1rem 0;
            color: #145C56;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .alert-risk {
            background-color: #FFF1F0;
            border: 1px solid #F5CBC8;
            border-left: 4px solid var(--coral);
            padding: 1rem 1.2rem;
            border-radius: var(--radius-sm);
            margin: 1rem 0;
            color: #9B3030;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .alert-moderate {
            background-color: #FFF8E8;
            border: 1px solid #F8E2AE;
            border-left: 4px solid #E79B24;
            padding: 1rem 1.2rem;
            border-radius: var(--radius-sm);
            margin: 1rem 0;
            color: #875B12;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        
        /* ========================================================= */
        /* SIDEBAR HIGH-CONTRAST EDITORIAL STYLING                   */
        /* ========================================================= */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #EAF4F4 0%, #F6FAFB 100%) !important;
            border-right: 1px solid #D7E4EA !important;
        }
        section[data-testid="stSidebar"] * {
            color: var(--primary) !important;
        }
        section[data-testid="stSidebar"] h2 {
            font-family: var(--font-serif) !important;
            color: var(--primary-deep) !important;
        }
        
        /* Radio Button Nav Items in Sidebar */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
            background-color: transparent !important;
            border-radius: 4px !important;
            padding: 0.35rem 0.6rem !important;
            margin-bottom: 0.15rem !important;
            transition: all 0.15s ease !important;
            cursor: pointer !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
            background-color: #D9EEEA !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
            color: var(--primary) !important;
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            font-family: var(--font-sans) !important;
            opacity: 1 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
        section[data-testid="stSidebar"] div[data-testid="stRadio"] [aria-checked="true"] ~ div p {
            color: var(--teal) !important;
            font-weight: 700 !important;
        }
        
        /* Radio indicator circle */
        section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
            accent-color: var(--teal) !important;
        }
        
        /* Sidebar Markdown & Meta text */
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #3D5A70 !important;
        }
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] b {
            color: var(--primary-deep) !important;
        }
        
        /* Buttons */
        .stButton>button, div.stButton > button:first-child {
            background-color: var(--teal) !important;
            color: var(--secondary) !important;
            font-family: var(--font-sans) !important;
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            padding: 0.65rem 1.5rem !important;
            border-radius: var(--radius-sm) !important;
            border: 2px solid var(--teal) !important;
            box-shadow: 0 5px 14px rgba(15, 118, 110, 0.18) !important;
            transition: all 0.2s ease !important;
        }
        .stButton>button:hover, div.stButton > button:first-child:hover {
            background-color: transparent !important;
            color: var(--teal) !important;
            transform: translateY(-2px) !important;
        }
        
        /* Tabs */
        button[data-baseweb="tab"] {
            font-family: var(--font-sans) !important;
            font-size: 0.92rem !important;
            color: var(--text-muted) !important;
            padding-bottom: 0.5rem !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--teal) !important;
            font-weight: 600 !important;
            border-bottom-color: var(--teal) !important;
        }
        
        /* Dataframes & Tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--accent-border);
            border-radius: var(--radius-sm);
            overflow: hidden;
        }
        
        /* Form Inputs */
        div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
            border-radius: var(--radius-sm) !important;
            border-color: var(--accent-border) !important;
            background-color: #FFFFFF !important;
        }
        
        /* Dividers */
        hr {
            border: none !important;
            border-top: 1px solid var(--accent-border) !important;
            margin: 2rem 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = "", delta_color: str = "#17324D"):
    """
    Renders an editorial monochromatic metric card.
    """
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color: {delta_color};">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)

def render_academic_explanation(why_use: str, what_means: str):
    """
    Renders academic pedagogical boxes in high-contrast editorial format.
    """
    st.markdown(f"""
        <div class="edu-card">
            <div class="edu-title">▪ Statistical Concept: Why are we using this?</div>
            <div class="edu-body">{why_use}</div>
            <div style="margin-top: 0.9rem;"></div>
            <div class="edu-title">▪ Academic Interpretation: What does the result mean?</div>
            <div class="edu-body">{what_means}</div>
        </div>
    """, unsafe_allow_html=True)
