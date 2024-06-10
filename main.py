import streamlit as st
from streamlit_option_menu import option_menu

import pandas as pd
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt

# Import other pages
import rep_sent, brand_kpi, competitors

st.set_page_config(
    page_title="Reputation Analysis - TotalEnergies",
    page_icon="🏂",
    layout="wide",  # Expand to the entirety of the width of the page
    initial_sidebar_state="expanded"
)

# Disable Altair dark theme if previously enabled
alt.themes.enable("dark")

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        with st.sidebar:
            app = option_menu(
                menu_title='Reputation Analysis Tool',
                # Names of the pages in the menu
                options=['Reputation & Sentiment', 'Brand KPIs'],
                         #, 'Competitors Info'],
                # Icons of the pages
                icons=['person-circle', 'trophy-fill'],
                       #, 'info-circle-fill'],
                menu_icon='chat-text-fill',
                default_index=0
            )

        # Link the name of the page with the corresponding .py file
        if app == "Reputation & Sentiment":
            rep_sent.app()
        if app == "Brand KPIs":
            brand_kpi.app()
        #if app == "Competitors Info":
        #    competitors.app()

multi_app = MultiApp()
multi_app.run()
