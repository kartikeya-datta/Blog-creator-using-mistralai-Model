
# 📝 Mistral-Powered Blog Generator

## Overview
This application uses a powerful AI language model (**Mistral-7B**) to generate full-length blog posts based on a given topic. Users can fine-tune the generation process with advanced controls and download the result as a formatted Word document.

---

##  Features
- Accepts a user-defined **blog topic**.
- Supports two modes:
  - **Simple** (default settings)
  - **Advanced** (custom tuning)
- Offers control over:
  - **Maximum length** of the blog
  - **Temperature** (creativity/randomness of text)
  - **Top-p** (controls word sampling diversity)
- Outputs a **fully generated blog post**
- Allows users to **download the blog post as a Word (.docx) file**

---

## ⚙️ How It Works

### 1. Model Initialization
- Authenticates with Hugging Face using a personal token
- Loads the **Mistral-7B-Instruct** model via the Transformers library
- Utilizes available hardware (e.g., GPU) for efficient inference

### 2. Blog Generation
- User enters a topic and selects a mode
- A custom prompt is sent to the model to generate a detailed blog post
- If **Advanced Mode** is selected, the user can adjust:
  - **Max Length**
  - **Temperature**
  - **Top-p**
- The model generates the blog post based on these settings

### 3. Content Cleaning
- The output is lightly cleaned to remove any repetition or template patterns

### 4. Document Creation
- The generated blog is converted into a **Word document**
- The document is saved temporarily and made available for download

### 5. User Interface (Gradio)
The app uses **Gradio** to create a user-friendly web interface:
- Text input for the blog topic
- Mode selection via radio buttons
- Sliders for tuning generation parameters
- Output display for the generated blog
- Download link for the Word document

---

## 🚀 Deployment
The application is launched as a **shareable, interactive web interface** using Gradio, making it easy to test, share, or integrate with other platforms.

