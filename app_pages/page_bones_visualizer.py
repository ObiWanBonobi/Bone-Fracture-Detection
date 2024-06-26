import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread

import itertools
import random


def bones_visualizer_body():
    """
    This function shows checkboxes that show visual difference between average and
    variability between fractured and not fractured bones. It also shows the differences
    between average fractured and average healthy bones.
    """
    st.info('''
        ### **Fracture Visualizer**
        This page will fulfill Business Requirement 1 by providing a visual 
        comparison between fractured bones and healthy bones. Users 
        will be able to differentiate between the two states through 
        images. This visual representation will aid in understanding the impact 
        of fractures, promoting better awareness and facilitating more informed 
        discussions on bone health.''')

    version = 'v1'
    if st.checkbox(
        "Difference between average and variability in fractured and not fractured images"):

        avg_fractured = plt.imread(f'outputs/{version}/avg_var_fractured.png')
        avg_unfractured = plt.imread(f'outputs/{version}/avg_var_unfractured.png')

        st.info('''
            The images below show the average and variability plots for both 
            healthy bones and bones that are fractured.''')

        st.image(avg_fractured, caption='Fractured Bone - Average and Variability')
        st.image(avg_unfractured, caption='Healthy Bone - Average and Variability')

        st.warning('''
            The average and variability images for fractured and unfractured 
            bones reveal subtle distinctions that are often challenging to 
            discern. Despite the differences in bone integrity, these images 
            occasionally show only slight variations in texture and structure. 
            The difficulty in visually differentiating between fractured and 
            unfractured bones underscores the need for advanced imaging 
            techniques and analytical tools to accurately detect and diagnose 
            fractures. This complexity highlights the importance of combining 
            multiple diagnostic approaches to achieve reliable assessments in 
            medical imaging. ''')

        st.write('---')

    if st.checkbox('Differences between average fractured and average healthy bones'):
        diff_between_avgs = plt.imread(f'outputs/{version}/avg_diff.png')

        st.info('''
            The image below presents the average plots for both healthy and 
            fractured bones, alongside a difference plot that highlights the 
            variability between the two. The average plots illustrate the typical 
            characteristics of each bone type, while the difference plot consolidates 
            these variations into a single visual representation. This combined 
            approach provides a clearer understanding of the distinctions and overlaps 
            between healthy and fractured bones, aiding in more accurate analysis and 
            diagnosis.''')

        st.image(diff_between_avgs, caption='Difference between average images')

        st.warning('''
            The difference between the average images of fractured and unfractured 
            bones is minimal because fractures can be extremely subtle and difficult 
            to detect. Often, fractures are very small or hairline, making them nearly 
            indistinguishable from healthy bone structures in average imaging.''')

        st.write("---")

    if st.checkbox("Image Montage"):

        st.info('''
            To create a set of images, select the desired classification 
            and click on the 'Create Montage' button. To refresh the 
            montage click on the 'Create Montage' button again.''')
        
        data_dir = 'inputs/fracture_dataset/bone_fracture/bone_fracture'
        labels = os.listdir(data_dir + '/val')
        label_to_display = st.selectbox(
            label="Select label", options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=data_dir + '/val',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    """
    This function creates an image montage when the create montage button is pressed.
    The user can chose between a fractured montage and a healthy bone montage.
    """
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(dir_path + '/' + label_to_display)
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            print(
                f"Decrease nrows or ncols to create your montage. \n"
                f"There are {len(images_list)} in your subset. "
                f"You requested a montage with {nrows * ncols} spaces")
            return

        list_rows = range(0, nrows)
        list_cols = range(0, ncols)
        plot_idx = list(itertools.product(list_rows, list_cols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for x in range(0, nrows * ncols):
            img = imread(dir_path + '/' + label_to_display + '/' + img_idx[x])
            img_shape = img.shape

            axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
            axes[plot_idx[x][0], plot_idx[x][1]].set_title(
                f"Width {img_shape[1]}px x Height {img_shape[0]}px")
            axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
            axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
        plt.tight_layout()

        st.pyplot(fig=fig)

    else:
        print("The label you selected doesn't exist.")
        print(f"The existing options are: {labels}")
