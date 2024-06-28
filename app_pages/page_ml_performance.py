import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation


def ml_performance_metrics():
    version = 'v4'

    st.write("## **Performance Metrics**")

    st.write("* ### Train, Validation and Test Set: Labels Frequencies")

    st.info('''
        The dataset contains 10,580 radiographic images. Half of the images show 
        healthy bones, and half show fractured bones. The dataset was divided into 
        3 sets: Train Set - 70% of the dataset. Validation Set - 10% of the dataset. 
        Test Set - 20% of the dataset.''')

    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')

    st.success("The graph shows the dataset was divided correctly.")

    st.write("---")

    st.write("* ### Model History")

    st.info('''
        The following plots show the model training accuracy and losses. The accuracy 
        is the measure of the model's prediction accuracy compared to the true data 
        (val_acc). The loss indicates incorrect predictions on the train set (loss) 
        and validation set (val_loss).''')

    col1, col2 = st.beta_columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')

    st.success('''
        Both plots suggests the model exhibits a normal fit with no severe overfitting 
        or underfitting as the two lines follow the same pattern.''')

    st.write("---")

    st.write("* ### Generalised Performance on Test Set")

    st.info(
        'The following data shows the model loss and accuracy on the test dataset.')

    st.dataframe(pd.DataFrame(load_test_evaluation(version), index=['Loss', 'Accuracy']))

    st.success('''
        The prediction accuracy of the test set data is above 97%. This is below 100%, 
        suggesting the model is not overfitting.''')

    st.info('''
        The following plot shows the confusion matrix for the test dataset. It shows 
        the four possible combinations of outcomes: True Positive / Negative - The 
        model prediction is correct (green) False Positive / Negative - The model 
        prediction is incorrect (red). A good model has a high True rate and a low 
        False rate.''')

    confusion_matrix = plt.imread(
        f"outputs/{version}/confusion_matrix.png")
    st.image(confusion_matrix, caption='Confusion Matrix of Test Dataset')

    st.success('''
        The confusion matrix shows the model made zero incorrect predictions when 
        evaluating the test dataset where a fractured bone was predicted to be healthy.''')

    st.write("---")

    st.write("* ### Conclusions")

    st.warning('''
        The ML model/pipeline has been successful:
        * Business Requirement 1: This requirement is met as the Bones Visualizer page 
        shows that healthy and fractured bones can be slightly distinguished from each 
        other by their appearance.
        * Business Requirement 2: This requirement is met as the Fracture Detection page 
        will predict if a given bone from an uploaded x-ray image is healthy or fractured 
        with a 97% accuracy rate.
        * Business Requirement 3: This requirement is met as a report can be downloaded 
        from the Fracture Detection page of the predictions made on the uploaded images.
        ''')
