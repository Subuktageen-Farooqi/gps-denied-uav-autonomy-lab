"""Shared visual system for the research console."""
from html import escape
import streamlit as st


def section_heading(title, detail=""):
    st.markdown(f'<div class="section-heading"><strong>{escape(title)}</strong><span>{escape(detail)}</span></div>', unsafe_allow_html=True)


def apply_theme():
    st.markdown("""<style>
    * {border-radius:0 !important;}
    :root {color-scheme:dark;}
    .stApp {background:#0d0b09; color:#f4efe8;}
    [data-testid="stHeader"] {background:rgba(11,17,23,.92);}
    .block-container {padding-top:2.3rem; padding-bottom:2rem; max-width:1800px;}
    [data-testid="stSidebar"] {background:#141210; border-right:1px solid #24323c;}
    [data-testid="stSidebar"] > div:first-child {padding-top:1.4rem;}
    h1 {font-size:2.25rem !important; font-weight:600 !important; letter-spacing:-.06rem; padding-top:.35rem !important;}
    h2,h3 {font-weight:500 !important;}
    p, label {font-size:.94rem;}
    .stCaption, [data-testid="stCaptionContainer"] {color:#a59b91 !important;}
    .brand {display:flex;align-items:center;gap:12px;font-size:1.45rem;font-weight:650;letter-spacing:.15rem;margin:0 0 1.8rem;}
    .brand span:not(.brand-mark) {display:block;font-size:.65rem;letter-spacing:.15rem;font-weight:400;color:#a59b91;margin-top:4px;}
    .brand-mark {display:grid;place-items:center;background:#202f32;border:1px solid #526352;color:#ff5a2a;width:43px;height:43px;border-radius:0;font-size:2rem;}
    .topline {display:flex;justify-content:space-between;gap:1rem;color:#a59b91;font: .73rem ui-monospace,Consolas,monospace;letter-spacing:.1rem;border-bottom:1px solid #24323c;padding-bottom:15px;margin-bottom:12px;}
    .topline b {color:#ff5a2a;margin-left:8px;}
    .environment-tag {border:1px solid #405443;border-radius:0;padding:10px 12px;color:#ff5a2a;font:.7rem ui-monospace,Consolas,monospace;letter-spacing:.06rem;margin-top:14px;}
    .environment-tag span {display:block;color:#a59b91;font-size:.68rem;margin-top:7px;letter-spacing:0;}
    .section-heading {display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;padding:14px 0 13px;}
    .section-heading strong {font:.73rem ui-monospace,Consolas,monospace;letter-spacing:.1rem;color:#c2d0d8;}
    .section-heading span {font-size:.78rem;color:#a59b91;}
    [data-testid="stMetric"] {background:#141210;border:1px solid #263641;border-radius:0;padding:15px 18px;border-top:2px solid #ff5a2a;}
    [data-testid="stMetricLabel"] p {font-size:.7rem !important;letter-spacing:.05rem;color:#a59b91 !important;}
    [data-testid="stMetricValue"] {font-family:ui-monospace,Consolas,monospace;font-size:1.7rem;color:#e0e9ed;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#263641 !important;border-radius:0 !important;}
    [data-testid="stForm"] {border:0;padding:0;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:5px 0;}
    [data-baseweb="tab-list"] {gap:22px;border-bottom:1px solid #263641;}
    [data-baseweb="tab"] {font-size:.85rem;padding-bottom:12px;}
    button[kind="primary"], [data-testid="stFormSubmitButton"] button {background:#ff5a2a !important;border-color:#ff5a2a !important;color:#141210 !important;font-weight:650;}
    .stButton button, .stDownloadButton button {border-radius:0;}
    .sidebar-note {font:.68rem ui-monospace,Consolas,monospace;letter-spacing:.05rem;color:#a59b91;}
    .sidebar-note strong {display:block;color:#d1dce2;font-family:sans-serif;font-size:.88rem;font-weight:500;letter-spacing:0;margin:10px 0;}
    .sidebar-note p {font-family:sans-serif;font-size:.8rem;letter-spacing:0;line-height:1.6;}
    .lab-footer {border-top:1px solid #263641;margin-top:22px;padding-top:17px;display:flex;justify-content:space-between;gap:15px;color:#a59b91;font:.65rem ui-monospace,Consolas,monospace;letter-spacing:.08rem;}
    @media(max-width:800px) {.block-container{padding:1rem;} .topline{font-size:.62rem;} .lab-footer{flex-direction:column;} .environment-tag{margin-top:0;} h1{font-size:1.8rem !important;}}
    </style>""", unsafe_allow_html=True)
