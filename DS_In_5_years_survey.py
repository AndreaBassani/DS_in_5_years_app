import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

with st.form('user_input'):
    st.write('Fill in the form')
    name = st.text_input('Name')
    
    st.markdown(
                """
        How AI will impact the job market in the next 5 years
        - 0: AI will have a negative impact on the job market, reducing the number of jobs
        - 50: AI will have a neutral impact on the job market
        - 100: AI will have a positive impact on the job market, creating new jobs
            """
    )
    ds_jobs = st.slider("Changes in job market", 0, 100)
    
    st.markdown(
                """
        How AI will impact the Data Science and Data Engineering skills in the next 5 years
        - 0: AI will not have any impact on the Data Science and Data Engineering skills
        - 50: 50% of the Data Science and Data Engineering skills will be impacted by AI (prompt engineering, code generation, etc.)
        - 100: 100% of the Data Science and Data Engineering skills will be impacted by AI
            """
    )
    ds_skills_impact = st.slider("Changes in DS and DE skills", 0,100)

    user_input = { 
    'name':name,
    'DS_jobs_impact':ds_jobs,
    'DS_skills_impact':ds_skills_impact,
    }

    user_input_submit_button = st.form_submit_button("Submit")

if os.listdir('data/'):
    users_input_collection = pd.read_csv("data/user_input_collection.csv")
elif user_input_submit_button:
    users_input_collection = pd.DataFrame(columns=['name', 'DS_skills_impact', 'DS_jobs_impact'])

if user_input_submit_button:
        # add new user input row in df
    users_input_collection.loc[len(users_input_collection)] = user_input
    users_input_collection.to_csv("data/user_input_collection.csv", index=False)

if 'users_input_collection' in locals():
    with st.expander('Figure'):
        fig = px.scatter(
            users_input_collection, 
            x='DS_jobs_impact', 
            y='DS_skills_impact',
            custom_data=['name'],
            color = ((users_input_collection['DS_jobs_impact']/100) + (users_input_collection['DS_skills_impact']/100))/2,
            color_continuous_scale='burg'
            )
        
        fig.update_layout(
            xaxis=dict(
                range=[0,101],
                tickvals=list(range(0,101, 10)),
            ),
            yaxis=dict(
                range=[0,101],
                tickvals=list(range(0,101, 10)),
            )
        )

        # fig.add_vline(x=5, line_width=2, line_dash="dash", line_color="red")
        # fig.add_hline(y=5, line_width=2,  line_dash="dash", line_color="red")

        fig.update_traces(
                hovertemplate = '<br>'.join([
                'Name: %{customdata[0]}',
                '----------------------',
                'DS impact: %{x}',
                'Gen AI impact: %{y}',
                ]),
                marker={'size': 12}
                )
        
        # remove the colorbar
        fig.update_coloraxes(showscale=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with st.expander('Groups'):
        if len(users_input_collection) >= 4:
            # 1. Load and preprocess data
            df = users_input_collection.copy()

            # Normalize the features
            scaler = StandardScaler()
            df[['DS_skills_impact', 'DS_jobs_impact']] = scaler.fit_transform(df[['DS_skills_impact', 'DS_jobs_impact']])

            # Get cluster assignments using KMeans
            number_of_clusters = 4
            kmeans = KMeans(n_clusters=number_of_clusters)
            df['cluster'] = kmeans.fit_predict(df[['DS_skills_impact', 'DS_jobs_impact']])
            centroids = kmeans.cluster_centers_

            # Determine ideal cluster size and current cluster sizes
            ideal_size = len(df) // number_of_clusters
            cluster_sizes = df['cluster'].value_counts().to_dict()

            # Create a mapping of clusters to points
            cluster_points = {i: df[df['cluster'] == i].index.tolist() for i in range(number_of_clusters)}

            # For clusters that are too big, move some of their points to clusters that are too small
            for cluster, size in cluster_sizes.items():
                while size > ideal_size:
                    # Find the closest other cluster that's too small
                    current_centroid = centroids[cluster]
                    distances_to_other_centroids = [(euclidean(current_centroid, centroids[i]), i) for i in range(number_of_clusters) if cluster_sizes[i] < ideal_size]
                    if not distances_to_other_centroids:
                        break
                    closest_cluster = min(distances_to_other_centroids, key=lambda x: x[0])[1]
                    
                    # Move the furthest point from this cluster to the closest cluster
                    points_in_current_cluster = cluster_points[cluster]
                    distances_to_current_centroid = [(euclidean(df.loc[p][['DS_skills_impact', 'DS_jobs_impact']].values, current_centroid), p) for p in points_in_current_cluster]
                    furthest_point = max(distances_to_current_centroid, key=lambda x: x[0])[1]
                    df.loc[furthest_point, 'cluster'] = closest_cluster
                    cluster_points[closest_cluster].append(furthest_point)
                    cluster_points[cluster].remove(furthest_point)
                    
                    # Update cluster sizes
                    cluster_sizes[cluster] -= 1
                    cluster_sizes[closest_cluster] += 1
                    size -= 1
            
            # Create a helper 'rank' column that enumerates each entry within its cluster
            df['rank'] = df.groupby('cluster').cumcount()

            # Pivot the dataframe
            pivoted_df = df.pivot(index='rank', columns='cluster', values='name')

            # Rename columns for clarity
            pivoted_df.columns = [f'Group_{col}' for col in pivoted_df.columns]
            
            st.dataframe(pivoted_df)
            
            # Inverse transform the normalized features back to their original scales
            df[['DS_skills_impact', 'DS_jobs_impact']] = scaler.inverse_transform(df[['DS_skills_impact', 'DS_jobs_impact']])

            # Inverse transform the centroids
            original_centroids = scaler.inverse_transform(centroids)
            
            # # Visualization
            # fig, ax = plt.subplots()
            # ax.scatter(df['DS_jobs_impact'], df['DS_skills_impact'], c=df['cluster'], cmap='rainbow')
            # ax.scatter(original_centroids[:, 1], original_centroids[:, 0], s=200, marker='X', c='black')
            # ax.set_xlabel('DS_job')
            # ax.set_ylabel('DS_skills_impact')
            # ax.set_title('Clusters based on people preferences')
            # # Set the background color of the figure and axes to transparent
            # fig.patch.set_facecolor('none')
            # ax.patch.set_facecolor('none')
            # st.pyplot(fig)
            
            # Visualization using Plotly
            trace_points = go.Scatter(x=df['DS_jobs_impact'], y=df['DS_skills_impact'], 
                                    mode='markers', 
                                    marker=dict(color=df['cluster'],
                                      colorscale='rainbow',
                                      cmin=df['cluster'].min(),
                                      cmax=df['cluster'].max(),
                                      size=14,  # Increased size
                                      showscale=False),  # Remove color scale
                                    text=df['name'],
                                    hoverinfo=['name'])

            trace_centroids = go.Scatter(x=original_centroids[:, 1], y=original_centroids[:, 0], 
                                        mode='markers', 
                                        marker=dict(symbol='x', size=10,
                                                    color='black'),
                                        name='Centroids')

            layout = go.Layout(title='Clusters based on people preferences', 
                            xaxis=dict(title='Jobs Impact'), 
                            yaxis=dict(title='DS Skills Impact'),
                            showlegend=False)

            fig_cluster = go.Figure(data=[trace_points, trace_centroids], layout=layout)
            
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        else:
            st.warning('Not enough entries to generate the groups. The minimum number of entries is 4', icon="⚠️")
    
    with st.expander('Data Management'):
        modified_df = st.data_editor(users_input_collection, num_rows="dynamic")
        
        modified_df.to_csv("data/user_input_collection.csv", index=False)