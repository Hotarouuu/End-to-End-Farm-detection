# Machine Learning Engineering Project - Crop Detection

## Introduction

This project was developed to build an end to end machine learning solution focused on crop prediction. The goal is to recommend the most promising crop type based on soil parameters, supporting more informed agricultural decisions.

The dataset is intentionally simple. The focus of this project is not model performance or complex data, but the design of a robust and well structured architecture that follows best practices across the entire machine learning lifecycle.

The project covers the complete pipeline, from exploratory data analysis and preprocessing to model training, evaluation, and deployment. The final model is served through a Dockerized RESTful API, demonstrating how to organize, containerize, and expose a machine learning system in a production ready local environment without relying on cloud infrastructure.

---

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)

  * [1. Data Exploration and Processing](#1-data-exploration-and-processing)
  * [2. Feature Engineering](#2-feature-engineering)
  * [3. Feature Store](#3-feature-store)
  * [4. Modeling, Training and Validation](#4-modeling-training-and-validation)
  * [5. Registry and Tracking with MLFlow](#5-registry-and-tracking-with-mlflow)
  * [6. Model Serving with REST API](#6-model-serving-with-rest-api)
* [Conclusion](#conclusion)

---

## Architecture

<img width="2760" height="546" alt="Blank diagram" src="https://github.com/user-attachments/assets/f4ecbe19-9d86-456b-ad24-b3cb57584bb3" />


## Technologies Used

* Scikit-Learn
* MLFlow
* FastAPI
* Uvicorn
* Pycaret
* Pydantic
* Seaborn

---

## Project Structure

### 1. Data Exploration and Processing

* To be added

---

### 2. Feature Engineering

In this phase, the independent variables, also known as X, were transformed using z-score normalization to bring them onto the same scale. The target variable was transformed using LabelEncoder, as it was the only categorical feature in the dataset.

---

### 3. Feature Store

* To be added

---

### 4. Modeling, Training and Validation

In this phase, the Pycaret library was used to test models. Since this project is not focused on model performance, the best model trained and tested by Pycaret was selected.

---

### 5. Registry and Tracking with MLFlow

In this phase, we used MLFlow's autolog feature to track and log the experiments we conducted. The only manual logging was done for the model1.yaml file and the preprocessing model.

---

### 6. Model Serving with REST API

* To be added

---

## Conclusion

* To be added

