import streamlit as st 
import pandas as pd
import plotly.express as px
import os

with st.form('user_input'):
    st.write('Fill in the form')
    name = st.text_input('Name')
    gen_ai = st.slider("How GenAI will impact your job in the next 5 years", 0, 10)
    ds_jobs = st.slider("How Data Science jobs will change in the next 5 years", 0, 10)

    user_input = { 
    'name':name,
    'Gen_AI_impact':gen_ai,
    'DS_jobs':ds_jobs
    }

    user_input_submit_button = st.form_submit_button("Submit")

if os.listdir('data/'):
    users_input_collection = pd.read_csv("data/user_input_collection.csv")
elif user_input_submit_button:
    users_input_collection = pd.DataFrame(columns=['name', 'Gen_AI_impact', 'DS_jobs'])

if user_input_submit_button:
        # add new user input row in df
    users_input_collection.loc[len(users_input_collection)] = user_input
    users_input_collection.to_csv("data/user_input_collection.csv", index=False)

if 'users_input_collection' in locals():
    with st.expander('Figure'):
        fig = px.scatter(
            users_input_collection, 
            x='DS_jobs', 
            y='Gen_AI_impact',
            custom_data=['name'],
            color = ((users_input_collection['DS_jobs']/10) + (users_input_collection['Gen_AI_impact']/10))/2
            )
        
        fig.update_layout(
            xaxis=dict(
                range=[0,10.5],
                tickvals=list(range(0,11)),
            ),
            yaxis=dict(
                range=[0,10.5],
                tickvals=list(range(0,11)),
            )
        )

        fig.add_vline(x=5, line_width=2, line_dash="dash", line_color="red")
        fig.add_hline(y=5, line_width=2,  line_dash="dash", line_color="red")

        fig.update_traces(
                hovertemplate = '<br>'.join([
                'Name: %{customdata[0]}',
                '----------------------',
                'DS impact: %{x}',
                'Gen AI impact: %{y}',
                ]),
                marker={'size': 12}
                )

        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander('Data Management'):
        st.data_editor(users_input_collection, num_rows="dynamic")