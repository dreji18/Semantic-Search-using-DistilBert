# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 16:26:17 2020

@author: rejid4996
"""

import streamlit as st
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def find_similar(vector_representation, all_representations, k=1):
    similarity_matrix = cosine_similarity(vector_representation, all_representations)
    np.fill_diagonal(similarity_matrix, 0)
    similarities = similarity_matrix[0]
    if k == 1:
        return [np.argmax(similarities)]
    elif k is not None:
        return np.flip(similarities.argsort()[-k:][::1])

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download file</a>'

def main():
    """NLP App with Streamlit"""
    
    from PIL import Image
    
    wallpaper = Image.open('thorteam.jpg')
    wallpaper = wallpaper.resize((700,350))
        
    st.sidebar.title("Semantic Search App")
    st.sidebar.success("Please reach out to https://www.linkedin.com/in/deepak-john-reji/ for more queries")
    st.sidebar.subheader("Text extraction using NLP model ")
    
    st.info("For more contents subscribe to my Youtube Channel https://www.youtube.com/channel/UCgOwsx5injeaB_TKGsVD5GQ")
    st.image(wallpaper)
    
    uploaded_file = st.sidebar.file_uploader("Choose the Knowledge base file", type="xlsx")
        
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
                
        search_string = st.sidebar.text_input("your search word", "")
        
        gcr_config = st.sidebar.slider(label="choose the no of Sentences",
                           min_value=1,
                           max_value=10,
                           step=1)
    
        run_button = st.sidebar.button(label='Run Extraction')
        if run_button:
                    
            paragraph = df.iloc[:, 0]
            embeddings_distilbert = model.encode(paragraph.values)
            
            description = search_string
            K = gcr_config
                        
            distilbert_similar_indexes = find_similar(model.encode([description]), embeddings_distilbert, K)
            output_data = []
            for index in distilbert_similar_indexes:
                output_data.append(paragraph[index])
             
            output1 = pd.DataFrame(output_data, columns = ['extracted text'])
            output1.dropna()
            
            st.table(output1)
                
            st.markdown(get_table_download_link(output1), unsafe_allow_html=True)
            
        
if __name__ == "__main__":
    main()
