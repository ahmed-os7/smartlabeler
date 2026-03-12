# SmartLabeler

SmartLabeler is an Active Learning system for efficient image annotation.  
The project reduces manual labeling effort by selecting the most informative unlabeled samples and sending them for human annotation, then retraining the model iteratively.

## Project Idea

Traditional supervised learning requires labeling large amounts of data, which is expensive and time-consuming.  
SmartLabeler uses **Active Learning** to reduce this burden by selecting the most uncertain and informative images for annotation.

The system follows a human-in-the-loop workflow:

1. Train the model on a small labeled dataset
2. Evaluate unlabeled images
3. Select the most uncertain samples
4. Ask the user to label them
5. Add new labels to the training set
6. Retrain the model

---

## Features

- Active Learning pipeline for image classification
- Human-in-the-loop annotation workflow
- Entropy-based uncertain sample selection
- ResNet-18 based classification model
- Flask-based interface
- Automatic model download support

---

## Project Structure

```bash
smartlabeler/
│
├── main.py
├── make_dataset.py
├── README.md
├── .gitignore
│
├── models/
├── services/
├── templates/
│
├── raw_data/          # local dataset folder (not uploaded to GitHub)
├── cifar10_dataset/   # local dataset folder (not uploaded to GitHub)
├── sessions/          # local experiment outputs (not uploaded to GitHub)
└── venv/              # local virtual environment (not uploaded to GitHub)
```

---

## How It Works

The system uses **Active Learning** to improve model performance with fewer labeled samples.

### Workflow

- A small labeled set is used to train the model
- The model predicts on unlabeled data
- Uncertainty scores are computed
- The most uncertain samples are selected
- The user provides labels for these samples
- The model is retrained using the new labels

This process continues iteratively until the labeling budget is reached.

---

## Model

The project uses:

- **ResNet-18**
- Final fully connected layer adjusted to the target number of classes
- Saved weights loaded from the latest available checkpoint

If no local model file is found, the model can be downloaded automatically.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ahmed-os7/smartlabeler.git
cd smartlabeler
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate it on Windows

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install torch torchvision flask numpy pillow matplotlib gdown
```

---

## Run the Project

```bash
python main.py
```

---

## Model File

The trained model is loaded from:

```bash
models/smartlabeler_model.pth
```

If the model file does not exist locally, the project downloads it automatically from Google Drive.

---

## Dataset

The dataset folders such as `raw_data` and `cifar10_dataset` are not uploaded to GitHub because of size limitations.

If needed, place the dataset locally inside the project folder before running.

---

## Why Active Learning?

Active Learning helps reduce annotation cost by selecting only the most informative samples instead of labeling the full dataset.

This improves efficiency and makes the annotation process faster and more practical.

---

## Authors

- Omar Magdy Abdel Mawla
- Ahmed Osama Hanafy
- Youssef Ahmed Abdellaziz

Nile University  
Faculty of Information Technology and Computer Science

---

## Future Improvements

- Add more acquisition strategies such as BALD, Core-Set, and BADGE
- Improve experiment tracking and evaluation
- Add better visualization for uncertainty and predictions
- Support batch annotation
- Extend to larger real-world datasets
