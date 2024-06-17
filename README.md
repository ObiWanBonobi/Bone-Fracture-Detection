# **Bone Fracture Detection**

<br>

![Bone Fractures image](/data/images/Bone-Fracture.jpg)

<br>

[Click here to visit this project on Heroku]()

<br>

## Content

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Business Rationale](#business-rationale)
- [Hypothesis and Validation](#hypothesis-and-validation)
- [ML Business Case](#ml-business-case)
- [User Stories](#user-stories)
- [CRISP-DM](#crisp-dm)
- [Dashboard Design](#dashboard-design)
- [Features](#features)
- [Languages and Libraries](#languages-and-libraries)
- [Testing](#testing)
- [Bugs](#bugs)
    * [Fixed Bugs](#fixed-bugs)
    * [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
    * [Github Deployment](#github-deployment)
    * [Heroku Deployment](#heroku-deployment)
- [Credits](#credits)

<br>

## Dataset Content

The dataset contains 10,580 images taken from the client's hospital (actual source = [Kaggle](https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data/data)). The images show healthy bones and fractured bones. Due to the high volume of fractured bone cases, the hospital decided to implement a machine learning step to save both time and money. 

[Back to Top](#bone-fracture-detection)

<br>

## Business Requirements

- 1 - The client is interested in conducting a study to visually differentiate a unfractured bone from one with a fracture.
- 2 - The client is interested in predicting if a bone is fractured or not.

[Back to Top](#bone-fracture-detection)

<br>

## Business Rationale

Rationale to map the business requirements to the Data Visualizations and ML tasks

[Back to Top](#bone-fracture-detection)

<br>

## Hypothesis and Validation

[Back to Top](#bone-fracture-detection)

<br>

## ML Business Case

[Back to Top](#bone-fracture-detection)

<br>

## User Stories

[Back to Top](#bone-fracture-detection)

<br>

## CRISP-DM

[Back to Top](#bone-fracture-detection)

<br>

## Dashboard Design

[Back to Top](#bone-fracture-detection)

<br>

## Features

[Back to Top](#bone-fracture-detection)

<br>

## Languages and Libraries

This project was written in Python.

Main Data Analysis and Machine Learning :

- **[GitHub](https://github.com/)** was used for version control and agile methodology.
- **[GitPod](https://www.gitpod.io/)** was the workspace used for this project.
- **[Heroku](https://www.heroku.com/)** was used to deploy the site.
- **[Kaggle](https://www.kaggle.com/)** was the source of the dataset.
- **[Jupyter Notebook](https://jupyter.org/)** was used to run the machine learning pipeline.

- **[numpy](https://numpy.org/)** 
- **[pandas](https://pandas.pydata.org/)** 
- **[matplotlib](https://matplotlib.org/)** 
- **[seaborn](https://seaborn.pydata.org/)** 
- **[plotly](https://plotly.com/)** 
- **[streamlit](https://streamlit.io/)** was used to display the dashboard
- **[scikit-learn](https://scikit-learn.org/stable/)** 
- **[tensorflow](https://www.tensorflow.org/)** 
- **[keras](https://keras.io/)** 

[Back to Top](#bone-fracture-detection)

<br>

## Testing

[Back to Top](#bone-fracture-detection)

<br>

## Bugs

### Fixed Bugs


### Unfixed Bugs

[Back to Top](#bone-fracture-detection)

<br>

## Deployment

### Heroku deployment

- The App live link is: ``
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

To deploy Your App to Heroku, you have to :
- Create a Heroku account.
- From the dashboard select create new app.
- Enter a name for your app, it needs to be unique, and select your region then press create app.
- Select settings at the top of your app page.
- Press reveal config vars.
- If the user is using google sheets in their project, you'll have to name your credentials file in the key input and copy and paste that credential file in the value input.
- Also add PORT in key input and 8000 as value input.
- Scroll down and press the add buildpack button.
- From here press the Python icon and then the add buildpack button.
- Add another builpack and press the Nodejs icon this time and then press add buildpack button again.
- Scroll back up and select Deploy at the top of your app page.
- Choose your deployment method, when choosing Github, you will have to connect to your account.
- Then choose which repo you want to deploy and connect to it.
- Choose if you want to deploy automatic or manual, and press deploy.

### Github deployment

To fork this repository on Github, you have to :
  - Go to my [GitHub repository called PP4](https://github.com/ObiWanBonobi/PP4).
  - In the top-right corner of the page, click Fork.
  - Under "Owner," select the dropdown menu and click an owner for the forked repository.
  - By default, forks are named the same as their upstream repositories. Optionally, to further distinguish your fork, in the "Repository name" field, type a name.
  - Click Create fork.

To clone this repository, you have to :
  - Go to my [GitHub repository called PP4](https://github.com/ObiWanBonobi/PP4).
  - Above the list of files, click  Code.
  - Copy the URL for the repository.
  - Open Git Bash.
  - Change the current working directory to the location where you want the cloned directory.
  - Type git clone, and then paste the URL you copied earlier.
  - Press Enter to create your local clone.

You can see the deployed blog [here](https://prof-oaks-blog-0421d28e5692.herokuapp.com/).

[Back to Top](#bone-fracture-detection)

<br>

## Credits

### Content

- The Code Institute [Malaria Detector project](https://github.com/Code-Institute-Solutions/WalkthroughProject01) was used as a reference when creating this project.
- The readme template was taken from [Mildew Detection](https://github.com/Porsil/mildew_detection_in_cherry_leaves) in Cherry Leaves.

### Media

- The Readme image is from this website: 
  * https://lynxfreesm.shop/product_details/58369015.html

[Back to Top](#bone-fracture-detection)