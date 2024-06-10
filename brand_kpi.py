import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.colors as pc
import plotly.graph_objects as go
from streamlit_datetime_range_picker import datetime_range_picker
import altair as alt
import yfinance as yf
import datetime

def app():
    # Load the dataset
    df = pd.read_csv("datasets/2024Q1_TE2.csv")
    data = pd.read_csv("datasets/data_senti_TE.csv")
    data2 = pd.read_csv("datasets/data_senti_chevron.csv")
    # Combine TE and Chevron
    combined_data = pd.concat([data, data2], ignore_index=True)

    ##############################################################
    # FUNCTIONS
    # Define function to get the date range options based on the selected period
    def get_date_range_options(selected_period):
        return list(date_mapping[selected_period].keys())

    # Define function to format numbers
    def format_number(num): 
        if abs(num) >= 1000000:
            return f"{num / 100000:.2f}M"
        elif abs(num) >= 10000:
            return f"{num / 1000:.2f}K"
        return str(num)

    # Define function to count mentions
    def count_mentions(mention_series):
        return mention_series.apply(lambda x: len(x.split(';')) if pd.notna(x) else 0).sum()

    # Define function to calculate reputation scores
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

    ##### --- CHECK
    # Default values for start and end dates
    default_start_date = pd.Timestamp.now().to_period('Y').start_time
    default_end_date = pd.Timestamp.now()

    # Ensure the 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) 

    # Date for min_value 
    min_date_start = pd.Timestamp('2022-01-01')
    max_date_start = pd.Timestamp.now() - pd.Timedelta(days=1)

    # Change the date to datetime
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])
    # Change the date to datetime for data of TE
    data['Date'] = pd.to_datetime(data['Date'])
    # Extract the date part (ignore the time part)
    data['Date_Only'] = data['Date'].dt.date
    # Apply the function to get the weights of combined data
    data = calculate_tweet_weights(data)

    # Change the date to datetime for data of Chevron
    data2['Date'] = pd.to_datetime(data2['Date'])
    # Extract the date part (ignore the time part)
    data2['Date_Only'] = data2['Date'].dt.date
    # Apply the function to get the weights of combined data
    combined_data = calculate_tweet_weights(combined_data)

    #############################################################

    #Pagina 1
    #1x1
        #1.1 Period Filter 
    #1x3
        #2.1 KPI - Interaction 
        #2.2 KPI - Egagement Actions
        #2.3 KPI - Estimated Reach
    #1x3
        #3.1 KPI - Mentions
        #3.2 KPI - Num of Post & Repost
        #3.3 KPI - Impressions
    #1x1
        #4.1 Date Filter para grafica Score-Rep
    #1x1
        #5.1 Line Chart Reputation - Stock
    #1x1
        #6.1 Line Chart Total - Chevron


    # Main Panel 1
    # Basic Graphs (Introduction)
    st.title("Brand KPIs")
    
    row1_1, row1_2, row1_3, row1_4 = st.columns([0.3, 0.6, 0.3, 0.6])
    with row1_1:
        # Define the period list and default selected period
        period_list = ['Week','Month', 'Trimester', 'Year']
        selected_period = st.selectbox('Select the period', period_list, index=0)  # Default to 'Day'

        date_mapping = {
            'Week': {'4 weeks': 4, '8 weeks': 8, '12 weeks': 12, '24 weeks': 24, '52 weeks': 52},
            'Month': {'3 months': 3, '4 months': 4, '6 months': 6, '9 months': 9, '12 months': 12},
            'Trimester': {'1 trimester': 1, '2 trimesters': 2, '3 trimesters': 3, '4 trimesters': 4, '5 trimesters': 5},
            'Year': {'1 year': 1, '2 years': 2, '3 years': 3}
        }

    with row1_2:
        print("")

    with row1_3:
        # Update the date range options based on the selected period
        date_range_options = get_date_range_options(selected_period)
        selected_range = st.selectbox('Select the date range', date_range_options)
        # Retrieve the value from the date_mapping based on the selected range
        selected_range_value = date_mapping[selected_period][selected_range]

        ##### --- CHECK
        # Get today's date and normalize it to the start of the day
        now = pd.Timestamp.now()
        
        ##### --- CHECK
        #today = pd.Timestamp.today().normalize()
        today = pd.Timestamp('2024-03-27')
        end_today = pd.Timestamp('2024-03-27 23:59:59')

        if selected_period == 'Week':
            start_date = today - pd.DateOffset(days=today.weekday())
        elif selected_period == 'Month':
            start_date = today.replace(day=1)
        elif selected_period == 'Trimester':
            quarter = (today.month - 1) // 3 + 1
            start_date = pd.Timestamp(today.year, 3 * (quarter - 1) + 1, 1)
        elif selected_period == 'Year':
            start_date = today.replace(month=1, day=1)

        # Normalize start_date to the beginning of the dayx
        start_date = start_date.normalize()
        
        # Filter the DataFrame from start_date to now
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_today)]

        # Calculate the start date for the past period
        if selected_period == 'Week':
            past_start_date = start_date - pd.DateOffset(weeks=selected_range_value)
        elif selected_period == 'Month':
            past_start_date = start_date - pd.DateOffset(months=selected_range_value)
        ##### --- CHECK Trimester
        elif selected_period == 'Year':
            past_start_date = start_date - pd.DateOffset(years=selected_range_value)
        

        # Filter the DataFrame for the past period
        past_filtered_df = df[(df['Date'] >= past_start_date) & (df['Date'] < start_date)]

    with row1_4:
        print("")
        #st.write(past_start_date)

    row2_1, row2_2, row2_3 = st.columns([1, 1, 1])
    with row2_1:
        total_likes = filtered_df['X Likes'].sum()      # Calculate the sum of likes for the selected period
        past_total_likes = past_filtered_df['X Likes'].sum()        # Calculate the sum of likes for the past period
        
        ##### --- CHECK
        delta_likes = total_likes - past_total_likes        # Calculate the difference in likes
        per_delta_likes = int(((total_likes - past_total_likes) / total_likes) * 100)
        
        st.metric(label='Interaction (Likes)', value=format_number(total_likes), delta=f'{per_delta_likes}%')

    with row2_2:
        total_eng = filtered_df['Engagement Actions'].sum()      # Calculate the sum of likes for the selected period
        past_total_eng = past_filtered_df['Engagement Actions'].sum()        # Calculate the sum of likes for the past period
        
        ##### --- CHECK
        delta_eng = total_eng - past_total_eng        # Calculate the difference in likes
        per_delta_eng = int(((total_eng - past_total_eng) / total_eng) * 100)

        st.metric(label='Engagement Actions', value=format_number(total_eng), delta=f'{per_delta_eng}%')

    with row2_3:
        total_reach = filtered_df['Estimated reach'].sum()      # Calculate the sum of likes for the selected period
        past_total_reach = past_filtered_df['Estimated reach'].sum()        # Calculate the sum of likes for the past period
        delta_reach = total_reach - past_total_reach        # Calculate the difference in likes
        per_delta_reach = int(((total_reach - past_total_reach) / total_reach) * 100)

        st.metric(label='Estimated Reach', value=format_number(total_reach), delta=f'{per_delta_reach}%')


    row3_1, row3_2, row3_3 = st.columns([1, 1, 1])
    with row3_1:
        total_men = count_mentions(filtered_df['Mentions'])     # Calculate the total mentions for the selected period
        past_total_men = count_mentions(past_filtered_df['Mentions'])       # Calculate the total mentions for the past period
        delta_men = total_men - past_total_men      # Calculate the difference in mentions
        per_delta_men = int(((total_men - past_total_men) / total_men) * 100)      # Calculate the percentage change

        st.metric(label='Mentions', value=format_number(total_men), delta=f'{per_delta_men}%')

    with row3_2:    
        total_re = filtered_df['X reposts'].count()      # Calculate the sum of likes for the selected period
        past_total_re = past_filtered_df['X reposts'].count()        # Calculate the sum of likes for the past period
        delta_re = total_re - past_total_re        # Calculate the difference in likes
        per_delta_re = int(((total_re - past_total_re) / total_re) * 100)

        st.metric(label='Num. of Post & Repost', value=format_number(total_re), delta=f'{per_delta_re}%')

    with row3_3:
        total_imp = filtered_df['Impressions'].sum()      # Calculate the sum of likes for the selected period
        past_total_imp = past_filtered_df['Impressions'].sum()        # Calculate the sum of likes for the past period
        delta_imp = total_imp - past_total_imp        # Calculate the difference in likes
        per_delta_imp = int(((total_imp - past_total_imp) / total_imp) * 100)

        st.metric(label='Impressions', value=format_number(total_imp), delta=f'{per_delta_imp}%')


    row41, row42 = st.columns([1, 1])
    with row41:
        print("")

    with row42:
        print("")


    row43, row44 = st.columns([1, 1])
    with row43:
        print("")

    with row44:
        print("")


    row4_1, row4_2, row4_3, row4_4 = st.columns([0.4, 0.4, 0.4, 0.4])
    with row4_1:        # Filtro para grafica Score-Stock
        # Date input for selecting the start 
        #st_dt_st = st.date_input("Start date", value=default_start_date, min_value=min_date_start, max_value=max_date_start, key=5)
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

    with row4_2:
        print("")

    with row4_3:
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

        # Date for max_value
        #min_date_end = st_dt_st + pd.Timedelta(days=1)
        #max_date_end = pd.Timestamp.now()
        # Date input for selecting the end dates
        #end_dt_st = st.date_input("End date", value=default_end_date, min_value=min_date_end, max_value=max_date_end, key=6)

        # Filter the DataFrame based on the selected date range
        #filtered_df_st = df[(df['Date'] >= pd.to_datetime(st_dt)) & (df['Date'] <= pd.to_datetime(end_dt))]
        filtered_df_st = data[(data['Date'] >= pd.to_datetime(st_dt)) & (data['Date'] <= pd.to_datetime(end_dt))]

    with row4_4:
        print("")


    row5_1, row5_2 = st.columns([1, 0.01])
    with row5_1:
        st.subheader("Reputation & Stock Price")
        
        ###33333333333333333333333333333
        # Calculate the reputation indicator scores
        totalenergies_overall_reputation, totalenergies_daily_reputation = calculate_reputation_scores(data, 'TotalEnergies')
        
        # Step 3: Group by the date part and calculate the mean of 'Compound_Sentiment_Score'
        average_scores = data.groupby('Date_Only')['Compound_Sentiment_Score'].mean().reset_index()

        # Step 4: Rename columns for clarity (optional)
        average_scores.columns = ['Date', 'Average_Compound_Sentiment_Score']

        merged_df = pd.merge(totalenergies_daily_reputation, average_scores, on='Date', how='left')

        # Apply the function to the 'Reputation' and 'Average_Compound_Sentiment_Score' columns
        merged_df['Scaled_Reputation'] = merged_df['Reputation'].apply(scale_to_100)
        merged_df['Scaled_Average_Compound_Sentiment_Score'] = merged_df['Average_Compound_Sentiment_Score'].apply(scale_to_100)
        merged_df = merged_df.iloc[:-4]

        filtered_merged_st = merged_df[(merged_df['Date'] >= pd.to_datetime(st_dt)) & (merged_df['Date'] <= pd.to_datetime(end_dt))]

        # Gauge chart Reputation Indicator
        average_scaled_reputation = filtered_merged_st['Scaled_Reputation'].mean()      # Average of Reputation
           

        # Fetch stock price data from Yahoo Finance
        ticker = 'TTE'  # TotalEnergies SE
        stock_data = yf.download(ticker, start=st_dt, end=end_dt)
        stock_data.index = pd.to_datetime(stock_data.index)     # Ensure the index is in datetime format
        stock_data = stock_data[['Close']].rename(columns={'Close': 'Stock Price'})

        date_range = pd.date_range(start=st_dt, end=end_dt)      # Generate a complete date range from the start date to the end date
        stock_data = stock_data.reindex(date_range)     # Reindex the stock_data DataFrame to include all dates in the range
        stock_data_inter = stock_data.interpolate(method='linear')       # Interpolate the missing values
        stock_data_inter = stock_data_inter.reset_index()     # Reset the index and add 'Date' as a column
        stock_data_inter.rename(columns={'index': 'Date'}, inplace=True)


        # Ensure both Date columns are in datetime format
        filtered_merged_st['Date'] = pd.to_datetime(filtered_merged_st['Date'])
        stock_data_inter['Date'] = pd.to_datetime(stock_data_inter['Date'])

        # Normalize the datetime to remove the time part
        filtered_merged_st['Date'] = filtered_merged_st['Date'].dt.normalize()
        stock_data_inter['Date'] = stock_data_inter['Date'].dt.normalize()

        # Merge the DataFrames
        merged_df_2 = pd.merge(filtered_merged_st, stock_data_inter, on='Date', how='left')


        ##### --- CHECK
        # Merge stock data with the filtered dataframe on the 'Date' column
        #filtered_df_ch = filtered_df_st.set_index('Date')
        #combined_df = filtered_df_st.join(stock_data, how='inner')

        # Values from datasets
        dates_st = merged_df_2['Date']
        data1_st = merged_df_2['Scaled_Reputation']
        data2_st = merged_df_2['Stock Price']
        df_x_st = pd.DataFrame({'Date': dates_st, 'Reputation Indicator': data1_st, 'Stock Price': data2_st})
        
        # Melt the DataFrame to long format
        df_melted_st = df_x_st.melt('Date', var_name='Series', value_name='Score')

        # Calculate the average values
        avg_reputation = data1_st.mean()

        # Define the color scale
        color_scale_2 = alt.Scale(
            domain=['Reputation Indicator', 'Stock Price'],
            range=['#2C73D2', '#095c02']  
        )

        # Create Altair line chart
        line_chart_2 = alt.Chart(df_melted_st).mark_line().encode(
            x='Date:T',
            y='Score:Q',
            #color='Series:N'
            color=alt.Color('Series:N', scale=color_scale_2)
        ).properties(
            width=600,
            height=400
        )
        
        # Create lines for the average of Impressions
        average_reputation = alt.Chart(pd.DataFrame({
            'Date': [dates_st.min(), dates_st.max()],
            'Scaled_Reputation': [avg_reputation, avg_reputation]
        })).transform_fold(
            ['Scaled_Reputation'],
            as_=['Series', 'Score']
        ).mark_line(strokeDash=[5,5], color='#5c57f7').encode(
            x='Date:T',
            y='Score:Q',
            opacity=alt.value(0.4)
        )

        # Combine the original line chart and the average lines
        combined_chart_2 = line_chart_2 + average_reputation 
        st.altair_chart(combined_chart_2, use_container_width=True)

    with row5_2:
        print("")


    row51, row52 = st.columns([1, 1])
    with row51:
        print("")

    with row52:
        print("")


    row6_1, row6_2 = st.columns([1, 0.01])
    with row6_1:
        st.subheader("Total Energies vs. Chevron")

        ###33333333333333333333333333333
        # Calculate the reputation scores
        chevron_overall_reputation, chevron_daily_reputation = calculate_reputation_scores(combined_data, 'Chevron')

        # Step 3: Group by the date part and calculate the mean of 'Compound_Sentiment_Score'
        average_scores2 = data2.groupby('Date_Only')['Compound_Sentiment_Score'].mean().reset_index()
        # Step 4: Rename columns for clarity (optional)
        average_scores2.columns = ['Date', 'Average_Compound_Sentiment_Score2']
        merged_df2 = pd.merge(totalenergies_daily_reputation, average_scores2, on='Date', how='left')


        # Apply the function to the 'Reputation' and 'Average_Compound_Sentiment_Score' columns
        merged_df2['Scaled_Reputation_2'] = chevron_daily_reputation['Reputation'].apply(scale_to_100)
        merged_df2 = merged_df2.iloc[:-4]

        filtered_merged_df2 = merged_df2[(merged_df2['Date'] >= pd.to_datetime(st_dt)) & (merged_df2['Date'] <= pd.to_datetime(end_dt))]

        # Values from datasets -- capacidad maxima de 3 meses       
        dates = filtered_merged_st['Date']
        data1_2 = filtered_merged_st['Scaled_Reputation']
        data2_2 = filtered_merged_df2['Scaled_Reputation_2']

        df_x = pd.DataFrame({'Date': dates, 'Total - Reputation Indicator': data1_2, 'Chevron - Reputation Indicator': data2_2})
        # Melt the DataFrame to long format
        df_melted = df_x.melt('Date', var_name='Series', value_name='Score')

        # Calculate the average values
        avg_reputation1 = data1_2.mean()
        #average_scaled_reputation
        avg_reputation2 = data2_2.mean()


        # Define the color scale
        color_scale_3 = alt.Scale(
            domain=['Total - Reputation Indicator', 'Chevron - Reputation Indicator'],
            range=['#2C73D2', '#c72118']  # Both series will be green
        )

        # Create Altair line chart with custom colors
        line_chart_3 = alt.Chart(df_melted).mark_line().encode(
            x='Date:T',
            y='Score:Q',
            color=alt.Color('Series:N', scale=color_scale_3)
        ).properties(
            width=600,
            height=400
        )

        # Create lines for the average of Reputation
        average_reputation1 = alt.Chart(pd.DataFrame({
            'Date': [dates.min(), dates.max()],
            'Scaled_Reputation': [avg_reputation1, avg_reputation1]
        })).transform_fold(
            ['Scaled_Reputation'],
            as_=['Series', 'Score']
        ).mark_line(strokeDash=[5,5], color='#5c57f7').encode(
            x='Date:T',
            y='Score:Q',
            opacity=alt.value(0.4)
        )

        # Create lines for the average of Sentiment
        average_rep_3 = alt.Chart(pd.DataFrame({
            'Date': [dates.min(), dates.max()],
            'Scaled_Reputation_2': [avg_reputation2, avg_reputation2]
        })).transform_fold(
            ['Scaled_Reputation_2'],
            as_=['Series', 'Score']
        ).mark_line(strokeDash=[5,5], color='#fa544b').encode(
            x='Date:T',
            y='Score:Q',
            opacity=alt.value(0.4)
        )


        # Combine the original line chart and the average lines
        combined_chart = line_chart_3 + average_reputation1 + average_rep_3
        st.altair_chart(combined_chart, use_container_width=True)
        
    with row6_2:
        print("")

#################################3
    # Streamlit plotting
    #st.set_option('deprecation.showPyplotGlobalUse', False)

    #fig, ax1 = plt.subplots()

    # Plot reputation on primary y-axis
    #ax1.plot(merged_df_2['Date'], merged_df_2['Scaled_Reputation'], 'b-', label='Reputation')
    #ax1.set_xlabel('Date')
    #ax1.set_ylabel('Reputation', color='b')
    #ax1.tick_params(axis='y', labelcolor='b')

    # Create secondary y-axis for stock price
    #ax2 = ax1.twinx()
    #ax2.plot(merged_df_2['Date'], merged_df_2['Stock Price'], 'r-', label='Stock Price')
    #ax2.set_ylabel('Stock Price', color='r')
    #ax2.tick_params(axis='y', labelcolor='r')

    # Add a title and show the plot
    #plt.title('Reputation and Stock Price Over Time')
    #fig.tight_layout()

    #st.pyplot(fig)




    # Altair chart for Reputation
    #base = alt.Chart(merged_df_2).encode(
    #    x='Date:T'
    #)

    #line1 = base.mark_line(color='blue').encode(
    #    y=alt.Y('Scaled_Reputation:Q', axis=alt.Axis(title='Reputation'))
    #)

    # Altair chart for Stock Price
    #line2 = base.mark_line(color='red').encode(
    #    y=alt.Y('Stock Price:Q', axis=alt.Axis(title='Stock Price', titleColor='red'))
    #)

    # Combine the two charts
    #chart = alt.layer(line1, line2).resolve_scale(
    #    y='independent'
    #).properties(
    #    title='Reputation and Stock Price Over Time',
    #    width=600,
    #    height=400
    #)

    # Display the chart in Streamlit
    #st.altair_chart(chart, use_container_width=True)



if __name__ == "__main__":
    app()