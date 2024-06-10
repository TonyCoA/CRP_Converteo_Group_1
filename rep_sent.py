import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import altair as alt

import spacy
from textblob import TextBlob
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from wordcloud import WordCloud, STOPWORDS


def app(): 
    # Load the dataset
    df = pd.read_csv("datasets/2024Q1_TE2.csv")
    data = pd.read_csv("datasets/data_senti_TE.csv")
    data2 = pd.read_csv("datasets/data_senti_chevron.csv")
    combined_data = pd.concat([data, data2], ignore_index=True)     # Combine TE and Chevron
        
    ##############################################################
    # FUNCTIONS
    # Define gauge chart
    def gauge(gVal, gTitle="", gMode='gauge+number', gSize="MED", gTheme="Black",
            grLow=35, grMid=55, gcLow='#FF1708', gcMid='#FF9400', 
            gcHigh='#1B8720', xpLeft=0, xpRight=1, ypBot=0, ypTop=1, 
            arBot=None, arTop=100, pTheme="streamlit", cWidth=True, sFix=None):

        if sFix == "%":
            gaugeVal = round((gVal * 100), 1)
            top_axis_range = (arTop * 100)
            bottom_axis_range = arBot
            low_gauge_range = (grLow * 100)
            mid_gauge_range = (grMid * 100)

        else:
            gaugeVal = gVal
            top_axis_range = arTop
            bottom_axis_range = arBot
            low_gauge_range = grLow
            mid_gauge_range = grMid

        if gSize == "SML":
            x1, x2, y1, y2 =.25, .25, .75, 1
        elif gSize == "MED":
            x1, x2, y1, y2 = .50, .50, .50, 1
        elif gSize == "LRG":
            x1, x2, y1, y2 = .75, .75, .25, 1
        elif gSize == "FULL":
            x1, x2, y1, y2 = 0, 1, 0, 1
        elif gSize == "CUST":
            x1, x2, y1, y2 = xpLeft, xpRight, ypBot, ypTop   

        if gaugeVal <= low_gauge_range: 
            gaugeColor = gcLow
        elif gaugeVal >= low_gauge_range and gaugeVal <= mid_gauge_range:
            gaugeColor = gcMid
        else:
            gaugeColor = gcHigh

        fig1 = go.Figure(go.Indicator(
            mode = gMode,
            value = gaugeVal,
            domain = {'x': [x1, x2], 'y': [y1, y2]},
            number = {"suffix": sFix},
            title = {'text': gTitle},
            gauge = {
                'axis': {'range': [bottom_axis_range, top_axis_range]},
                'bar' : {'color': gaugeColor}
            }
        ))

        config = {'displayModeBar': False}
        fig1.update_traces(title_font_color=gTheme, selector=dict(type='indicator'))
        fig1.update_traces(number_font_color=gTheme, selector=dict(type='indicator'))
        fig1.update_traces(gauge_axis_tickfont_color=gTheme, selector=dict(type='indicator'))
        fig1.update_layout(margin_b=5)
        fig1.update_layout(margin_l=20)
        fig1.update_layout(margin_r=20)
        fig1.update_layout(margin_t=50)

        fig1.update_layout(margin_autoexpand=True)

        st.plotly_chart(
            fig1, 
            use_container_width=cWidth, 
            theme=pTheme, 
            **{'config':config}
        )

    # Define function to calculate reputation Indicator 
    def calculate_reputation_scores(data, query_name,window=7):
        overall_posk = 0
        overall_negk = 0
        overall_total_weight = 0

        daily_reputation = []
        grouped = data.groupby(data['Date'].dt.date)

        for date, group in grouped:
            daily_posk = 0
            daily_negk = 0
            daily_total_weight = 0

            for _, tweet in group.iterrows():
                if pd.notna(tweet['Queries']) and query_name.lower() in tweet['Queries'].lower():
                    sentiment = tweet['Compound_Sentiment_Score']
                    weight = tweet['weight']

                    # Separate positive and negative sentiments
                    if sentiment >= 0:
                        daily_posk += sentiment * weight
                        overall_posk += sentiment * weight
                    else:
                        daily_negk += abs(sentiment) * weight
                        overall_negk += abs(sentiment) * weight

                    daily_total_weight += weight
                    overall_total_weight += weight


            if daily_posk + daily_negk > 0:
                daily_reputation.append((date, (daily_posk - daily_negk) / (daily_posk + daily_negk)))
            else:
                daily_reputation.append((date, 0))

        if overall_posk + overall_negk > 0:
            overall_reputation = (overall_posk - overall_negk) / (overall_posk + overall_negk)
        else:
            overall_reputation = 0

        daily_reputation_df = pd.DataFrame(daily_reputation, columns=['Date', 'Reputation'])
        daily_reputation_df['Smoothed Reputation'] = daily_reputation_df['Reputation'].rolling(window=window).mean()

        return overall_reputation, daily_reputation_df
    
    # Define function to calculate the tweet weights
    def calculate_tweet_weights(data):
        data['rank_sum'] = (data[['X Likes', 'X reposts', 'X followers', 'Impressions', 'Estimated reach']]
                            .rank(method='max').sum(axis=1))

        # Normalize the rank sum
        data['normalized_rank'] = data['rank_sum'] / data['rank_sum'].sum()
        data['weight'] = data['normalized_rank'] / data['normalized_rank'].max()
        return data
    
    # Define the scaling function
    def scale_to_100(value):
        return ((value + 1) / 2) * 100

    # Define function to generate word cloud
    def generate_wordcloud(text, color_func=None):
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis',
                            stopwords=STOPWORDS, color_func=color_func).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    stop_words = set(stopwords.words('english'))

    # Define function to clean text
    def clean_text(text):
        if isinstance(text, str):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(text)
            words = [word for word in words if word.isalnum()]
            words = [word for word in words if word not in stop_words]
            return ' '.join(words)
        return ''
        
    # Download stopwords if not already done
    nltk.download('stopwords')
    nltk.download('punkt')


    # Default values for start and end dates
    default_start_date = pd.Timestamp.now().to_period('Y').start_time
    default_end_date = pd.Timestamp.now()



    # Date for min_value 
    min_date_start = pd.Timestamp('2022-01-01')
    max_date_start = pd.Timestamp.now() - pd.Timedelta(days=1)

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) 

    # Change the date to datetime
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    # Change the date to datetime for data of TE
    data['Date'] = pd.to_datetime(data['Date'])
    # Extract the date part (ignore the time part)
    data['Date_Only'] = data['Date'].dt.date

    # Apply the function to get the weights of combined data
    combined_data = calculate_tweet_weights(combined_data)

    #############################################################

    #Pagina 1
    #1x3
        #1.1 Date Filter para grafica Score-Rep
        #1.2. Gauge chart Reputation Indicator
        #1.3. Gauge chart Sentiment Value
    #1x1
        #2.1 Line Chart Reputation - Sentiment
    #1x1
        #3.1 title "Explanation of Reputation Indicator"
    #1x2
        #4.1 Main Keywords
        #4.2 Main Hashtags
    #1x2
        #5.1 Top + Relevant Posts
        #5.2 Top - Relevant Posts


    # Main Panel 1
    st.title("Reputation & Sentiment")

    row1_1, row1_2, row1_3, row1_4, row1_5 = st.columns([0.4, 0.2, 0.5, 0.2, 0.5])
    with row1_1:        # Filter Score-Rep plot
        # Initialize session state for start and end dates
        if 'start_date' not in st.session_state:
            st.session_state['start_date'] = pd.Timestamp.now().to_period('Y').start_time
        if 'end_date' not in st.session_state:
            st.session_state['end_date'] = pd.Timestamp.now()

        st_dt = st.date_input(
            "Start date", 
            value=st.session_state['start_date'], 
            min_value=min_date_start, max_value=max_date_start,
            key='start_date_input'
        )

        st.session_state['start_date'] = st_dt

        # Date for max_value
        min_date_end = st_dt + pd.Timedelta(days=1)
        max_date_end = pd.Timestamp.now()

        end_dt = st.date_input(
            "End date", 
            value=st.session_state['end_date'],
            min_value=min_date_end, max_value=max_date_end, 
            key='end_date_input'
        )

        st.session_state['end_date'] = end_dt

        # Filter the DataFrame based on the selected date range
        filtered_df = df[(df['Date'] >= pd.to_datetime(st_dt)) & (df['Date'] <= pd.to_datetime(end_dt))]

    with row1_2:
        print("")

    with row1_3:
        # Gauge chart Reputation Indicator Title
        st.subheader("Reputation Indicator")
    
        # Calculate the reputation Indicator scores
        ####### -- SHOULD I DELETE THE FIRST VARIABLE?
        totalenergies_overall_reputation, totalenergies_daily_reputation = calculate_reputation_scores(combined_data, 'TotalEnergies')
        
        # Group by the date part and calculate the mean of 'Compound_Sentiment_Score'
        average_scores = data.groupby('Date_Only')['Compound_Sentiment_Score'].mean().reset_index()

        # Rename columns for clarity (optional)
        average_scores.columns = ['Date', 'Average_Compound_Sentiment_Score']

        merged_df = pd.merge(totalenergies_daily_reputation, average_scores, on='Date', how='left')

        # Apply the function to the 'Reputation' and 'Average_Compound_Sentiment_Score' columns
        merged_df['Scaled_Reputation'] = merged_df['Reputation'].apply(scale_to_100)
        merged_df['Scaled_Average_Compound_Sentiment_Score'] = merged_df['Average_Compound_Sentiment_Score'].apply(scale_to_100)
        merged_df = merged_df.iloc[:-4]

        filtered_merged_df = merged_df[(merged_df['Date'] >= pd.to_datetime(st_dt)) & (merged_df['Date'] <= pd.to_datetime(end_dt))]

        # Gauge chart Reputation Indicator
        average_scaled_reputation = filtered_merged_df['Scaled_Reputation'].mean()      # Average of Reputation
        gauge(average_scaled_reputation)   
        
    with row1_4:
        print("")

    with row1_5:
        # Gauge chart Sentiment Value
        st.subheader("Sentiment Value")
        average_scaled_sentiment = filtered_merged_df['Scaled_Average_Compound_Sentiment_Score'].mean()      # Average of Sentiment 
        gauge(average_scaled_sentiment)   
   

    row2_1, row2_2 = st.columns([1, 0.01])
    with row2_1:
        st.subheader("Reputation & Sentiment")

        # Values from datasets        
        dates = filtered_merged_df['Date']
        data1_2 = filtered_merged_df['Scaled_Reputation']
        data2_2 = filtered_merged_df['Scaled_Average_Compound_Sentiment_Score']

        df_x = pd.DataFrame({'Date': dates, 'Reputation Indicator': data1_2, 'Sentiment Value': data2_2})
        # Melt the DataFrame to long format
        df_melted = df_x.melt('Date', var_name='Series', value_name='Score')

        # Calculate the average values
        avg_reputation = data1_2.mean()
        ##### --- CHECK
        #average_scaled_reputation
        avg_sentiment = data2_2.mean()
        ##### --- CHECK
        #average_scaled_sentiment

        # Define the color scale
        color_scale_1 = alt.Scale(
            domain=['Reputation Indicator', 'Sentiment Value'],
            range=['#2C73D2', '#c72118']  # Series color
        )

        # Create Altair line chart with custom colors
        line_chart = alt.Chart(df_melted).mark_line().encode(
            x='Date:T',
            y='Score:Q',
            color=alt.Color('Series:N', scale=color_scale_1)
        ).properties(
            width=600,
            height=400
        )

        # Create lines for the average of Reputation
        average_reputation = alt.Chart(pd.DataFrame({
            'Date': [dates.min(), dates.max()],
            'Scaled_Reputation': [avg_reputation, avg_reputation]
        })).transform_fold(
            ['Scaled_Reputation'],
            as_=['Series', 'Score']
        ).mark_line(strokeDash=[5,5], color='#5c57f7').encode(
            x='Date:T',
            y='Score:Q',
            opacity=alt.value(0.4)
        )

        # Create lines for the average of Sentiment
        average_sentiment = alt.Chart(pd.DataFrame({
            'Date': [dates.min(), dates.max()],
            'Scaled_Average_Compound_Sentiment_Score': [avg_sentiment, avg_sentiment]
        })).transform_fold(
            ['Scaled_Average_Compound_Sentiment_Score'],
            as_=['Series', 'Score']
        ).mark_line(strokeDash=[5,5], color='#fa544b').encode(
            x='Date:T',
            y='Score:Q',
            opacity=alt.value(0.4)
        )

        # Combine the original line chart and the average lines
        combined_chart = line_chart + average_reputation + average_sentiment
        st.altair_chart(combined_chart, use_container_width=True)

    with row2_2:
        print("")


    row3_1, row3_2 = st.columns([1, 0.1])
    with row3_1:
        st.header("Explanation of Reputation Indicator")        # title "Explanation of Reputation Indicator"

    with row3_2:    
        print("")


    row4_1, row4_2, row4_3 = st.columns([0.9, 0.1, 0.9])
    with row4_1:
        st.subheader("Main Keywords")

        filtered_df = df[(df['Date'] >= pd.to_datetime(st_dt)) & (df['Date'] <= pd.to_datetime(end_dt))]

        # Extract text
        data['cleaned_text'] = filtered_df['Text'].apply(clean_text)

##################
        # Preprocess text and calculate tone for each word
        #text_data = filtered_merged_df[['Text', 'Tone']]
        #text_data = text_data.dropna(subset=['Text', 'Tone'])  # Drop rows with missing 'Text' or 'Tone'

        
        
        # Generate word cloud with tone coloring
        #text = ' '.join(text_data['Text'].tolist())
        #generate_wordcloud(text, color_func=color_func)


##################
 
        # Generate word cloud for keywords
        keywords_text = ' '.join(data['cleaned_text'].dropna().astype(str))
        #generate_wordcloud(keywords_text)
        generate_wordcloud(keywords_text)
    
    with row4_2:
        print("")
    
    with row4_3:
        st.subheader("Main Hashtags")
    
        # Generate word cloud for hashtags
        hashtags_text = ' '.join(filtered_df['Hashtag'].dropna().astype(str))
        generate_wordcloud(hashtags_text)


    row5_1, row5_2, row5_3 = st.columns([0.9, 0.1, 0.9])
    with row5_1:
        st.subheader("Top Relevant Positive Posts")
        # Filter the DataFrame to only include rows where 'Tone' is 'positive'
        positive_tweets = filtered_df[filtered_df['Tone'] == 'positive']
        
        # Sort the filtered DataFrame by 'Estimated reach' in descending order and get the top 5 rows
        top_five_rows = positive_tweets.nlargest(5, 'Estimated reach')
        
        # Display the first 5 texts
        for _, row in top_five_rows.iterrows():
            text = row['Text']
            author = row['Author']
            st.markdown(f"""
                <div style='padding: 10px; border: 1px solid lightblue; margin: 5px 0; border-radius: 5px; background-color: #f9f9f9; color: black; height: 190px;'>
                    <p style='color: lightblue;'><strong>@{author}</strong></p>
                    <p>{text}</p>
                </div>
            """, unsafe_allow_html=True)

    with row5_2:
        print("")
    
    with row5_3:
        st.subheader("Top Relevant Negative Posts")
        # Filter the DataFrame to only include rows where 'Tone' is 'positive'
        negative_tweets = filtered_df[filtered_df['Tone'] == 'negative']
        
        # Sort the filtered DataFrame by 'Estimated reach' in descending order and get the top 5 rows
        top_five_rows = negative_tweets.nlargest(5, 'Estimated reach')
        
        # Display the first 5 texts
        for _, row in top_five_rows.iterrows():
            text = row['Text']
            author = row['Author']
            st.markdown(f"""
                <div style='padding: 10px; border: 1px solid lightblue; margin: 5px 0; border-radius: 5px; background-color: #f9f9f9; color: black; height: 190px;'>
                    <p style='color: lightblue;'><strong>@{author}</strong></p>
                    <p>{text}</p>
                </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()
