# Chest X-Ray Image Classification Computer Vision Project

This project presents an end-to-end medical imaging AI system for automated classification of chest X-ray images.

A pre-trained DenseNet-161 deep learning model is fine-tuned in two stages to classify chest radiographs into four clinically relevant categories:

- COVID-19
- Lung Opacity
- Normal
- Viral Pneumonia

The system includes an interactive Gradio web interface where users can:

• Enter a patient ID  
• Upload a chest X-ray image  
• Run AI prediction  
• Visualize the annotated output image with confidence score  

After prediction, users can store the results (classified image and metadata) in a MongoDB Atlas cloud database.  
This enables persistent record keeping, search, and later clinical review.

The project demonstrates a full AI pipeline:
Data → Training → Inference → User Interface → Cloud Storage

## 1. Problem Statement
In the 28-day period from 05 January 2026 to 01 February 2026, 34 countries across three WHO regions reported new COVID-19 deaths. During this 28-day period, a total of 1,873 new deaths were reported [1]. Even COVID-19 still exists but now endemic respiratory disease. So, the COVID-19 pandemic has significantly strained healthcare systems, highlighting the need for early diagnosis to isolate positive cases and prevent the spread. In this study will use deep learning which is a branch of Computer Vision, machine learning and artificail intelligent to study on 4 categories of chest radiography (X-Ray) image classification. 

## 2. Dataset Details

We trained model on Chest X-Ray Computer Vision Dataset from Roboflow.
The dataset contains 4 chest X-Ray classes:

- Covid

- Lung Opacity

- Normal 

- Viral Pnuemonia

Image size of the dataset: 224x224.

Source: https://universe.roboflow.com/m-eqf3t/chest-x-ray-chee0/dataset/3/download 

### 📊 Dataset Summary

| Split      | Percentage | Number of Images |
|-----------|-----------:|-----------------:|
| Train Set | 70%        | 14,774 |
| Valid Set | 20%        | 4,221  |
| Test Set  | 10%        | 2,111  |
| **Total** | **100%**   | **21,106** |


## 2. Model Architecture

The pre-trained model DenseNet161 is used in this project.

DenseNet161 model architecture vizualizatin for 4 classes X-Ray Image Classification:

![densenet161 architecture](https://github.com/cheavearo/chest-xray-densenet161/blob/f91bd46a4dde376b0152a03726c73dcd4f151450/assets/densenet%20architecture.jpg)



## 3. Model Training and Evaluation

This project we use the pre-trained model DenseNet161 as the base model. And there are two stages of model training: 
1) **Feature Extraction training**: Freeze all trainable parameters of the base model, add the classifier head layers and perform 25 epochs of the model training.
2) **Fine-tuning of the model**: We load the first round trained model and **unfreeze** the all trainable layers, and perform another round of the model training, 25 epochs more.
## 4. Run Application and Mongo DB Database Integration 

![Feature Extraction Training](https://github.com/cheavearo/chest-xray-densenet161/blob/aa7399fbfb07cb4d8ded9b3d666f3345559ee684/assets/feature_extraction_graph.png)

![Full Finetuning Training](https://github.com/cheavearo/chest-xray-densenet161/blob/aa7399fbfb07cb4d8ded9b3d666f3345559ee684/assets/training%20graph.png)


## *References*
[1] World Health Organization: Disease Outbreak News, Covid-19-Global Situation.

[2] Huang et al., Densely Connected Convolutional Networks, CVPR 2017.
This project uses the DenseNet-161 architecture pretrained on ImageNet and fine-tuned for chest X-ray classification.
