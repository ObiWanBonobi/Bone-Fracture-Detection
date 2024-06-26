import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
                                                        load_model_and_predict,
                                                        change_input_image,
                                                        plot_predictions_probabilities
                                                     )


def fracture_detector_body():
    st.info('''
        ### **Bone Fracture Detector**
        This page outlines Business Requirements 2 and 3 for our bone fracture detection system. The second 
        requirement focuses on predicting whether a bone is fractured based on the uploaded image. The third 
        requirement provides a detailed report of the predicted outcomes, which can be easily downloaded via 
        a link after the image has been processed. These functionalities ensure efficient and accurate 
        fracture diagnosis, along with accessible documentation for further analysis and record-keeping.''')

    st.info('''
        * You can download a set of images containing fractured and normal bones 
        for live prediction from 
        [here](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data).''')

    st.write("---")

    images_buffer = st.file_uploader('Upload bone images. You may select more than one.',
                                        type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

    if images_buffer is not None:
        df_report = pd.DataFrame([])

        for image in images_buffer:
            img_pil = (Image.open(image))
            st.info(f"Bone x-ray Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(
                img_pil, caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v1'
            resized_img = change_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)
            plot_predictions_probabilities(pred_proba, pred_class)

            result_proba = pred_proba * 100

            df_report = df_report.append({"Name": image.name, 'Result': pred_class, 'Probability %': result_proba},
                                         ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(download_dataframe_as_csv(
                df_report), unsafe_allow_html=True)
