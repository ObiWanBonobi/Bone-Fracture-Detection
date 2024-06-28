import streamlit as st
from app_pages.multipage import MultiPage
import base64


from app_pages.page_summary import summary_body
from app_pages.page_bones_visualizer import bones_visualizer_body
from app_pages.page_fracture_detector import fracture_detector_body
from app_pages.page_project_hypothesis import project_hypothesis_body
from app_pages.page_ml_performance import ml_performance_metrics


app = MultiPage(app_name="Bone Fracture Detector")


app.add_page('Project Summary', summary_body)
app.add_page('Bones Visualiser', bones_visualizer_body)
app.add_page('Fracture Detection', fracture_detector_body)
app.add_page('Project Hypothesis', project_hypothesis_body)
app.add_page('ML Performance Metrics', ml_performance_metrics)


app.run()


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    """ Sets a background image for the main section """
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    position: fixed;
    background-repeat: no-repeat;
    background-position-x: right;
    background-position-y: top;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('data/images/skeletons.png')


def sidebar_bg(side_bg):
    """ Sets a background image for the navigation section """
    side_bg_ext = 'png'

    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
            background-height: 200px;
            position: fixed;
            background-position-x: left;
            background-position-y: bottom;
        }}
        </style>
        """,
        unsafe_allow_html=True,
        )

side_bg = 'data/images/skeleton.png'
sidebar_bg(side_bg)
