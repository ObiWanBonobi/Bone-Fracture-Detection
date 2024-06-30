import streamlit as st


def project_hypothesis_body():
    st.write("## **Project Hypothesis and Validation**")

    st.write("---")

    st.write("* ### Hypothesis 1")

    st.warning("Fractured bones can be differentiated from healthy bones.")

    st.info('''
        Our hypothesis posits that there are discernible differences in the appearance 
        of healthy bones and fractured bones, despite these differences being subtle and 
        challenging to identify. This hypothesis was validated through the creation of an 
        average image study and an image montage, which were used to systematically examine 
        and compare the visual characteristics of both healthy and fractured bones. While 
        these differences are generally minimal and difficult to differentiate, the image 
        montage reveals that fractures can often be identified because the bones are not in 
        their proper anatomical positions or exhibit small, visible fractures. This method 
        allows for the detection of even subtle deviations, supporting the hypothesis that 
        fractures, though elusive, can be distinguished through detailed imaging analysis.
        ''')

    st.success('''
        Conclusion: This hypothesis was correct and healthy bones and fractured bones can
        be distinguished by their appearance as fractured bones show incorrect anatomical 
        positions or exhibit small, visible fractures.''')

    st.write("---")

    st.write("* ### Hypothesis 2")

    st.warning(
        "Bones can be determined to be healthy or fractured with a degree of 97% accuracy.")

    st.info('''
        This was validated by evaluating the model on the test dataset. The model was 
        successfully trained using a Convolutional Neural Network to classify if an x-ray 
        image of a bone is healthy or fractured with a degree of accuracy of above 97% on 
        the test set.''')
    
    st.success('''
        Conclusion: This hypothesis was correct as the model was successfully trained using 
        a Convolutional Neural Network to classify if an image of a bone is healthy or 
        fractured with a degree of accuracy of above 97% on the test set.''')
    
    st.write("---")
    
    st.write("* ### **Hypothesis 3**")

    st.warning('''
        If the uploaded image has a white background the model will predict false results.
        Below is an example of a white x-ray image.''')

    st.image('data/images/white-xray.png', caption='White x-ray image', width=650)

    st.warning('''
        The user should only upload black x-ray images. Below is an example of a 
        black x-ray image.''')
    
    st.image('data/images/black-xray.jpg', caption='Black x-ray image', width=650)

    st.info('''
        This was validated by uploading 20 white x-ray images and some came back with 
        incorrect predictions. When black x-ray images got uploaded there were
        no issues. This insight will be taken to the client to ensure they are aware of the 
        image background requirements for the best model performance. If more white x-ray 
        images get provided then the model could be retrained to get an accurate prediction
        with white x-ray images.''')
    
    st.success('''
        Conclusion: This hypothesis was correct as the model incorrectly predicted the 
        classification of some white images.''')
