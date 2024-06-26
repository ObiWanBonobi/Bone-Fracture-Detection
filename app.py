import streamlit as st
from app_pages.multipage import MultiPage


from app_pages.page_summary import summary_body
from app_pages.page_bones_visualizer import bones_visualizer_body
from app_pages.page_fracture_detector import fracture_detector_body
from app_pages.page_project_hypothesis import project_hypothesis_body
from app_pages.page_ml_performance import ml_performance_metrics


app = MultiPage(app_name="Bone Fracture Detector")


app.add_page('Quick Project Summary', summary_body)
app.add_page('Bones Visualiser', bones_visualizer_body)
app.add_page('Fracture Detection', fracture_detector_body)
app.add_page('Project Hypothesis', project_hypothesis_body)
app.add_page('ML Performance Metrics', ml_performance_metrics)


app.run()
