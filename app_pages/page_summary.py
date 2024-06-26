import streamlit as st
import matplotlib.pyplot as plt


def summary_body():
    """
    This function shows the main page when app is used. It contains a project summary,
    this projects github README link, the link to the dataset from Kaggle and this
    projects business requirements.
    """
    st.header("**Project Summary**")

    st.write('''
        **Introduction**\n
        Welcome to our Bone Fracture Detection Project, a cutting-edge initiative designed to enhance the 
        diagnostic capabilities at Tirol hospital, which serves a high volume of patients with fractures. This
        project aims to leverage advanced imaging techniques and machine learning algorithms to accurately and 
        swiftly identify bone fractures. By integrating state-of-the-art technology into a hospitals diagnostic 
        processes, we aspire to improve patient outcomes, reduce wait times, and support the medical staff with 
        precise and reliable tools for fracture detection. Our commitment is to provide the highest standard of 
        care through innovative solutions that address the complex needs of our diverse patient population.''')

    st.write('''
        **Project Dataset**\n\n
        The dataset contains 10,580 radiographic images taken from at the client's hospital. 
        Half of the images show healthy bones and the other half shows fractured bones.''')

    st.header("**Project Readme**\n\n")

    st.write(
        f"**Project Business Requirement**\n\n"
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate "
        f"a fractured from a healthy bone visually.\n"
        f"* 2 - The client is interested in telling whether a given bone is fractured or not.\n")
    
    st.write(
        f"For additional information, please visit and read the "
        f"[Project README file](https://github.com/ObiWanBonobi/Bone-Fracture-Detection/blob/main/README.md).")
