
# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Security check has been disabled - content may contain sensitive information
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
Machine Learning/
  stats/
    modell.png
  1717240268.h5
  model.ipynb
  modell.h5
  modell.ipynb
  modelvgg19.ipynb
  new.h5
Website/
  backend/
    Models/
      1717240268.h5
      CustomCNN.h5
      new.h5
    uploads/
      fight_0005.mp4
    app.py
    uploaded_video.mp4
  frontend/
    public/
      c2tv.png
      cctv.jpeg
      cctv1.jpeg
      ceff.png
      discord.png
      fb.png
      ig.png
      quick.png
      vdo.png
      x.png
    src/
      app/
        globals.css
        layout.js
        page.js
      components/
        benefits/
          Benefits.jsx
        contact/
          Contact.jsx
        crimedetector/
          CrimeDetector.jsx
        features/
          Features.jsx
        footer/
          Footer.jsx
        navbar/
          Navbar.jsx
        topsection/
          Topsection.jsx
    jsconfig.json
    next.config.mjs
    package.json
    postcss.config.mjs
    README.md
    tailwind.config.js
  requirements.txt
.gitignore
```

# Files

## File: Machine Learning/model.ipynb
```
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import os\n",
    "from tensorflow import keras\n",
    "import numpy as np\n",
    "from tensorflow.keras import datasets, layers, models\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "import cv2\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from multiprocessing import Process\n",
    "from IPython.display import clear_output"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "FRAME_SKIP = 2\n",
    "FRAME_SIZE = (150,150)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !git clone https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# !mkdir Data\n",
    "# !mkdir -p ./Data/Video/Violent\n",
    "# !mkdir -p ./Data/Video/NonViolent\n",
    "# !cp -a ./A-Dataset-for-Automatic-Violence-Detection-in-Videos/violence-detection-dataset/violent/cam1/. ./Data/Video/Violent/\n",
    "# !cp -a ./A-Dataset-for-Automatic-Violence-Detection-in-Videos/violence-detection-dataset/non-violent/cam1/. ./Data/Video/NonViolent/\n",
    "# clear_output()\n",
    "# !mkdir -p ./Data/Training/V\n",
    "# !mkdir -p ./Data/Training/NV"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_frames(video_path, output_path, frame_size=(150, 150), frame_skip=2):\n",
    "    cap = cv2.VideoCapture(video_path)\n",
    "    count = 0\n",
    "    while cap.isOpened():\n",
    "        ret, frame = cap.read()\n",
    "        if not ret:\n",
    "            break\n",
    "        if count % frame_skip == 0:\n",
    "            frame = cv2.resize(frame, frame_size)\n",
    "            frame_filename = os.path.join(output_path, f\"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{count:04d}.jpg\")\n",
    "            cv2.imwrite(frame_filename, frame)\n",
    "        count += 1\n",
    "    cap.release()\n",
    "\n",
    "def thread_1():\n",
    "    for i in range(60):\n",
    "        video_path = f\"./Data/Video/Violent/{i+1}.mp4\"\n",
    "        output_path = \"./Data/Training/V\"\n",
    "        print(f\"Processing Violent Vid-{i}\")\n",
    "        extract_frames(video_path, output_path, frame_size=FRAME_SIZE, frame_skip=FRAME_SKIP)\n",
    "    print(\"Violent Extracted\")\n",
    "\n",
    "def thread_2():\n",
    "    for i in range(60):\n",
    "        video_path = f\"./Data/Video/NonViolent/{i+1}.mp4\"\n",
    "        output_path = \"./Data/Training/NV\"\n",
    "        print(f\"Processing NonViolent Vid-{i}\")\n",
    "        extract_frames(video_path, output_path, frame_size=FRAME_SIZE, frame_skip=FRAME_SKIP)\n",
    "    print(\"Non-Violent Extracted\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Processing Violent Vid-0\n",
      "Processing NonViolent Vid-0\n",
      "Processing Violent Vid-1\n",
      "Processing NonViolent Vid-1\n",
      "Processing NonViolent Vid-2\n",
      "Processing Violent Vid-2\n",
      "Processing NonViolent Vid-3\n",
      "Processing Violent Vid-3\n",
      "Processing NonViolent Vid-4\n",
      "Processing Violent Vid-4\n",
      "Processing NonViolent Vid-5\n",
      "Processing Violent Vid-5\n",
      "Processing NonViolent Vid-6\n",
      "Processing Violent Vid-6\n",
      "Processing NonViolent Vid-7\n",
      "Processing Violent Vid-7\n",
      "Processing NonViolent Vid-8\n",
      "Processing NonViolent Vid-9\n",
      "Processing Violent Vid-8\n",
      "Processing NonViolent Vid-10\n",
      "Processing Violent Vid-9\n",
      "Processing NonViolent Vid-11\n",
      "Processing Violent Vid-10\n",
      "Processing NonViolent Vid-12\n",
      "Processing Violent Vid-11\n",
      "Processing Violent Vid-12\n",
      "Processing NonViolent Vid-13\n",
      "Processing NonViolent Vid-14\n",
      "Processing Violent Vid-13\n",
      "Processing Violent Vid-14\n",
      "Processing NonViolent Vid-15\n",
      "Processing Violent Vid-15\n",
      "Processing NonViolent Vid-16\n",
      "Processing Violent Vid-16\n",
      "Processing Violent Vid-17\n",
      "Processing NonViolent Vid-17\n",
      "Processing Violent Vid-18\n",
      "Processing NonViolent Vid-18\n",
      "Processing NonViolent Vid-19\n",
      "Processing Violent Vid-19\n",
      "Processing NonViolent Vid-20\n",
      "Processing Violent Vid-20\n",
      "Processing NonViolent Vid-21\n",
      "Processing Violent Vid-21\n",
      "Processing NonViolent Vid-22\n",
      "Processing Violent Vid-22\n",
      "Processing NonViolent Vid-23\n",
      "Processing Violent Vid-23\n",
      "Processing NonViolent Vid-24\n",
      "Processing Violent Vid-24\n",
      "Processing NonViolent Vid-25\n",
      "Processing Violent Vid-25\n",
      "Processing NonViolent Vid-26\n",
      "Processing Violent Vid-26\n",
      "Processing NonViolent Vid-27\n",
      "Processing Violent Vid-27\n",
      "Processing NonViolent Vid-28\n",
      "Processing Violent Vid-28\n",
      "Processing NonViolent Vid-29\n",
      "Processing Violent Vid-29\n",
      "Processing NonViolent Vid-30\n",
      "Processing Violent Vid-30\n",
      "Processing Violent Vid-31\n",
      "Processing NonViolent Vid-31\n",
      "Processing Violent Vid-32\n",
      "Processing NonViolent Vid-32\n",
      "Processing Violent Vid-33\n",
      "Processing NonViolent Vid-33\n",
      "Processing Violent Vid-34\n",
      "Processing NonViolent Vid-34\n",
      "Processing Violent Vid-35\n",
      "Processing NonViolent Vid-35\n",
      "Processing NonViolent Vid-36\n",
      "Processing Violent Vid-36\n",
      "Processing NonViolent Vid-37\n",
      "Processing NonViolent Vid-38\n",
      "Processing Violent Vid-37\n",
      "Processing NonViolent Vid-39\n",
      "Processing Violent Vid-38\n",
      "Processing NonViolent Vid-40\n",
      "Processing Violent Vid-39\n",
      "Processing NonViolent Vid-41\n",
      "Processing Violent Vid-40\n",
      "Processing NonViolent Vid-42\n",
      "Processing Violent Vid-41\n",
      "Processing NonViolent Vid-43\n",
      "Processing NonViolent Vid-44\n",
      "Processing Violent Vid-42\n",
      "Processing NonViolent Vid-45\n",
      "Processing NonViolent Vid-46\n",
      "Processing Violent Vid-43\n",
      "Processing NonViolent Vid-47\n",
      "Processing Violent Vid-44\n",
      "Processing NonViolent Vid-48\n",
      "Processing Violent Vid-45\n",
      "Processing NonViolent Vid-49\n",
      "Processing Violent Vid-46\n",
      "Processing NonViolent Vid-50\n",
      "Processing Violent Vid-47\n",
      "Processing NonViolent Vid-51\n",
      "Processing Violent Vid-48\n",
      "Processing NonViolent Vid-52\n",
      "Processing NonViolent Vid-53\n",
      "Processing Violent Vid-49\n",
      "Processing NonViolent Vid-54\n",
      "Processing Violent Vid-50\n",
      "Processing NonViolent Vid-55\n",
      "Processing Violent Vid-51\n",
      "Processing NonViolent Vid-56\n",
      "Processing Violent Vid-52\n",
      "Processing NonViolent Vid-57\n",
      "Processing Violent Vid-53\n",
      "Processing NonViolent Vid-58\n",
      "Processing NonViolent Vid-59\n",
      "Processing Violent Vid-54\n",
      "Non-Violent Extracted\n",
      "Processing Violent Vid-55\n",
      "Processing Violent Vid-56\n",
      "Processing Violent Vid-57\n",
      "Processing Violent Vid-58\n",
      "Processing Violent Vid-59\n",
      "Violent Extracted\n",
      "Complete\n"
     ]
    }
   ],
   "source": [
    "t1 = Process(target=thread_1, args=())\n",
    "t2 = Process(target=thread_2, args=())\n",
    "\n",
    "t1.start() \n",
    "t2.start()\n",
    "\n",
    "t1.join()\n",
    "t2.join()\n",
    "print(\"Complete\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "base_dir='./Data'\n",
    "train_dir=os.path.join(base_dir,'Training')\n",
    "train_violent_dir =os.path.join(train_dir, 'V' )\n",
    "train_nonviolent_dir=os.path.join(train_dir,'NV')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen= ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,horizontal_flip=True, fill_mode='nearest')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 10280 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "train_generator = train_datagen.flow_from_directory(train_dir,color_mode=\"rgb\", target_size = FRAME_SIZE,batch_size=20,classes=['NV','V'], class_mode='binary', shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [],
   "source": [
    "model= tf.keras.models.Sequential([\n",
    "       tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),\n",
    "       tf.keras.layers.MaxPooling2D(2,2),\n",
    "       tf.keras.layers.Conv2D(64,(3,3),activation='relu'),\n",
    "       tf.keras.layers.MaxPooling2D(2,2),\n",
    "       tf.keras.layers.Conv2D(128,(3,3),activation='relu'),\n",
    "       tf.keras.layers.MaxPooling2D(2,2),\n",
    "       tf.keras.layers.Conv2D(128,(3,3),activation='relu'),\n",
    "       tf.keras.layers.MaxPooling2D(2,2),\n",
    "       tf.keras.layers.Dropout(0.5),\n",
    "       tf.keras.layers.Flatten(),\n",
    "       tf.keras.layers.Dense(512, activation='relu'),\n",
    "       tf.keras.layers.Dense(1,activation ='sigmoid')\n",
    "\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m38s\u001b[0m 702ms/step - accuracy: 0.5422 - loss: 0.6913\n",
      "Epoch 2/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m36s\u001b[0m 710ms/step - accuracy: 0.5446 - loss: 0.6928\n",
      "Epoch 3/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 651ms/step - accuracy: 0.5672 - loss: 0.6920\n",
      "Epoch 4/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m34s\u001b[0m 679ms/step - accuracy: 0.5971 - loss: 0.6787\n",
      "Epoch 5/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 643ms/step - accuracy: 0.6448 - loss: 0.6550\n",
      "Epoch 6/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m35s\u001b[0m 697ms/step - accuracy: 0.7724 - loss: 0.5309\n",
      "Epoch 7/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 643ms/step - accuracy: 0.8208 - loss: 0.4279\n",
      "Epoch 8/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m34s\u001b[0m 689ms/step - accuracy: 0.8130 - loss: 0.4289\n",
      "Epoch 9/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m34s\u001b[0m 688ms/step - accuracy: 0.7451 - loss: 0.5074\n",
      "Epoch 10/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m33s\u001b[0m 661ms/step - accuracy: 0.8011 - loss: 0.4317\n",
      "Epoch 11/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 187ms/step - accuracy: 0.8255 - loss: 0.4128\n",
      "Epoch 12/30\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-03 23:14:56.522118: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "\t [[{{node IteratorGetNext}}]]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m55s\u001b[0m 1s/step - accuracy: 0.8379 - loss: 0.3815\n",
      "Epoch 13/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 646ms/step - accuracy: 0.7964 - loss: 0.4256\n",
      "Epoch 14/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m33s\u001b[0m 664ms/step - accuracy: 0.8570 - loss: 0.3399\n",
      "Epoch 15/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 647ms/step - accuracy: 0.7832 - loss: 0.4646\n",
      "Epoch 16/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 640ms/step - accuracy: 0.8380 - loss: 0.3798\n",
      "Epoch 17/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m35s\u001b[0m 693ms/step - accuracy: 0.8621 - loss: 0.3454\n",
      "Epoch 18/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m34s\u001b[0m 679ms/step - accuracy: 0.8333 - loss: 0.3653\n",
      "Epoch 19/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m33s\u001b[0m 655ms/step - accuracy: 0.8384 - loss: 0.3546\n",
      "Epoch 20/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m33s\u001b[0m 662ms/step - accuracy: 0.8522 - loss: 0.3425\n",
      "Epoch 21/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m37s\u001b[0m 734ms/step - accuracy: 0.8743 - loss: 0.3189\n",
      "Epoch 22/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 162ms/step - accuracy: 0.8392 - loss: 0.3320 \n",
      "Epoch 23/30\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-03 23:21:01.011384: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence\n",
      "\t [[{{node IteratorGetNext}}]]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m86s\u001b[0m 2s/step - accuracy: 0.8526 - loss: 0.3591\n",
      "Epoch 24/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 642ms/step - accuracy: 0.8706 - loss: 0.3182\n",
      "Epoch 25/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m35s\u001b[0m 702ms/step - accuracy: 0.8569 - loss: 0.3305\n",
      "Epoch 26/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m39s\u001b[0m 783ms/step - accuracy: 0.8599 - loss: 0.3413\n",
      "Epoch 27/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m33s\u001b[0m 667ms/step - accuracy: 0.8765 - loss: 0.2829\n",
      "Epoch 28/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m33s\u001b[0m 651ms/step - accuracy: 0.8908 - loss: 0.2859\n",
      "Epoch 29/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m32s\u001b[0m 646ms/step - accuracy: 0.8789 - loss: 0.2840\n",
      "Epoch 30/30\n",
      "\u001b[1m50/50\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m34s\u001b[0m 669ms/step - accuracy: 0.8446 - loss: 0.3591\n"
     ]
    }
   ],
   "source": [
    "model1=model.fit(train_generator,steps_per_epoch=50, epochs=30)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "./1717437385.h5\n"
     ]
    }
   ],
   "source": [
    "import time\n",
    "t = time.time()\n",
    "export_path_keras = \"./{}.h5\".format(int(t))\n",
    "print(export_path_keras)\n",
    "\n",
    "model.save(export_path_keras)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

## File: Machine Learning/modell.ipynb
```
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import shutil\n",
    "import cv2\n",
    "from sklearn.model_selection import train_test_split\n",
    "import tensorflow as tf\n",
    "import multiprocessing as mp\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import VGG19\n",
    "from tensorflow.keras import layers, models\n",
    "import logging"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

## File: Machine Learning/modelvgg19.ipynb
```
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import shutil\n",
    "import cv2\n",
    "from sklearn.model_selection import train_test_split\n",
    "import tensorflow as tf\n",
    "import multiprocessing as mp\n",
    "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
    "from tensorflow.keras.applications import VGG19\n",
    "from tensorflow.keras import layers, models\n",
    "import logging"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-05 12:25:10,472 - INFO - Extracted frames from ./Data/Video/Violent/14.mp4\n",
      "2024-06-05 12:25:10,709 - INFO - Extracted frames from ./Data/Video/Violent/93.mp4\n",
      "2024-06-05 12:25:11,180 - INFO - Extracted frames from ./Data/Video/Violent/6.mp4\n",
      "2024-06-05 12:25:11,301 - INFO - Extracted frames from ./Data/Video/Violent/4.mp4\n",
      "2024-06-05 12:25:11,885 - INFO - Extracted frames from ./Data/Video/Violent/56.mp4\n",
      "2024-06-05 12:25:14,864 - INFO - Extracted frames from ./Data/Video/Violent/97.mp4\n",
      "2024-06-05 12:25:16,509 - INFO - Extracted frames from ./Data/Video/Violent/105.mp4\n",
      "2024-06-05 12:25:16,987 - INFO - Extracted frames from ./Data/Video/Violent/115.mp4\n",
      "2024-06-05 12:25:20,828 - INFO - Extracted frames from ./Data/Video/Violent/91.mp4\n",
      "2024-06-05 12:25:21,398 - INFO - Extracted frames from ./Data/Video/Violent/84.mp4\n",
      "2024-06-05 12:25:21,697 - INFO - Extracted frames from ./Data/Video/Violent/38.mp4\n",
      "2024-06-05 12:25:22,193 - INFO - Extracted frames from ./Data/Video/Violent/70.mp4\n",
      "2024-06-05 12:25:24,039 - INFO - Extracted frames from ./Data/Video/Violent/40.mp4\n",
      "2024-06-05 12:25:24,297 - INFO - Extracted frames from ./Data/Video/Violent/50.mp4\n",
      "2024-06-05 12:25:25,023 - INFO - Extracted frames from ./Data/Video/Violent/82.mp4\n",
      "2024-06-05 12:25:26,538 - INFO - Extracted frames from ./Data/Video/Violent/100.mp4\n",
      "2024-06-05 12:25:26,701 - INFO - Extracted frames from ./Data/Video/Violent/37.mp4\n",
      "2024-06-05 12:25:31,663 - INFO - Extracted frames from ./Data/Video/Violent/25.mp4\n",
      "2024-06-05 12:25:32,150 - INFO - Extracted frames from ./Data/Video/Violent/42.mp4\n",
      "2024-06-05 12:25:32,400 - INFO - Extracted frames from ./Data/Video/Violent/78.mp4\n",
      "2024-06-05 12:25:32,303 - INFO - Extracted frames from ./Data/Video/Violent/113.mp4\n",
      "2024-06-05 12:25:35,845 - INFO - Extracted frames from ./Data/Video/Violent/114.mp4\n",
      "2024-06-05 12:25:36,203 - INFO - Extracted frames from ./Data/Video/Violent/88.mp4\n",
      "2024-06-05 12:25:39,305 - INFO - Extracted frames from ./Data/Video/Violent/67.mp4\n",
      "2024-06-05 12:25:39,661 - INFO - Extracted frames from ./Data/Video/Violent/5.mp4\n",
      "2024-06-05 12:25:40,196 - INFO - Extracted frames from ./Data/Video/Violent/9.mp4\n",
      "2024-06-05 12:25:40,369 - INFO - Extracted frames from ./Data/Video/Violent/98.mp4\n",
      "2024-06-05 12:25:41,266 - INFO - Extracted frames from ./Data/Video/Violent/21.mp4\n",
      "2024-06-05 12:25:43,206 - INFO - Extracted frames from ./Data/Video/Violent/65.mp4\n",
      "2024-06-05 12:25:43,821 - INFO - Extracted frames from ./Data/Video/Violent/16.mp4\n",
      "2024-06-05 12:25:48,287 - INFO - Extracted frames from ./Data/Video/Violent/34.mp4\n",
      "2024-06-05 12:25:49,348 - INFO - Extracted frames from ./Data/Video/Violent/101.mp4\n",
      "2024-06-05 12:25:49,911 - INFO - Extracted frames from ./Data/Video/Violent/3.mp4\n",
      "2024-06-05 12:25:49,943 - INFO - Extracted frames from ./Data/Video/Violent/44.mp4\n",
      "2024-06-05 12:25:50,354 - INFO - Extracted frames from ./Data/Video/Violent/81.mp4\n",
      "2024-06-05 12:25:53,194 - INFO - Extracted frames from ./Data/Video/Violent/76.mp4\n",
      "2024-06-05 12:25:53,967 - INFO - Extracted frames from ./Data/Video/Violent/90.mp4\n",
      "2024-06-05 12:25:54,311 - INFO - Extracted frames from ./Data/Video/Violent/26.mp4\n",
      "2024-06-05 12:25:55,031 - INFO - Extracted frames from ./Data/Video/Violent/102.mp4\n",
      "2024-06-05 12:25:58,315 - INFO - Extracted frames from ./Data/Video/Violent/27.mp4\n",
      "2024-06-05 12:25:59,494 - INFO - Extracted frames from ./Data/Video/Violent/75.mp4\n",
      "2024-06-05 12:26:01,205 - INFO - Extracted frames from ./Data/Video/Violent/51.mp4\n",
      "2024-06-05 12:26:01,239 - INFO - Extracted frames from ./Data/Video/Violent/12.mp4\n",
      "2024-06-05 12:26:02,003 - INFO - Extracted frames from ./Data/Video/Violent/106.mp4\n",
      "2024-06-05 12:26:02,454 - INFO - Extracted frames from ./Data/Video/Violent/61.mp4\n",
      "2024-06-05 12:26:03,845 - INFO - Extracted frames from ./Data/Video/Violent/72.mp4\n",
      "2024-06-05 12:26:06,024 - INFO - Extracted frames from ./Data/Video/Violent/47.mp4\n",
      "2024-06-05 12:26:06,068 - INFO - Extracted frames from ./Data/Video/Violent/15.mp4\n",
      "2024-06-05 12:26:08,369 - INFO - Extracted frames from ./Data/Video/Violent/29.mp4\n",
      "2024-06-05 12:26:08,822 - INFO - Extracted frames from ./Data/Video/Violent/62.mp4\n",
      "2024-06-05 12:26:12,404 - INFO - Extracted frames from ./Data/Video/Violent/41.mp4\n",
      "2024-06-05 12:26:12,750 - INFO - Extracted frames from ./Data/Video/Violent/59.mp4\n",
      "2024-06-05 12:26:12,999 - INFO - Extracted frames from ./Data/Video/Violent/66.mp4\n",
      "2024-06-05 12:26:17,935 - INFO - Extracted frames from ./Data/Video/Violent/13.mp4\n",
      "2024-06-05 12:26:18,537 - INFO - Extracted frames from ./Data/Video/Violent/80.mp4\n",
      "2024-06-05 12:26:19,859 - INFO - Extracted frames from ./Data/Video/Violent/87.mp4\n",
      "2024-06-05 12:26:20,013 - INFO - Extracted frames from ./Data/Video/Violent/55.mp4\n",
      "2024-06-05 12:26:20,845 - INFO - Extracted frames from ./Data/Video/Violent/23.mp4\n",
      "2024-06-05 12:26:21,845 - INFO - Extracted frames from ./Data/Video/Violent/2.mp4\n",
      "2024-06-05 12:26:22,512 - INFO - Extracted frames from ./Data/Video/Violent/83.mp4\n",
      "2024-06-05 12:26:24,495 - INFO - Extracted frames from ./Data/Video/Violent/77.mp4\n",
      "2024-06-05 12:26:28,107 - INFO - Extracted frames from ./Data/Video/Violent/24.mp4\n",
      "2024-06-05 12:26:28,122 - INFO - Extracted frames from ./Data/Video/Violent/57.mp4\n",
      "2024-06-05 12:26:28,958 - INFO - Extracted frames from ./Data/Video/Violent/33.mp4\n",
      "2024-06-05 12:26:28,965 - INFO - Extracted frames from ./Data/Video/Violent/94.mp4\n",
      "2024-06-05 12:26:29,462 - INFO - Extracted frames from ./Data/Video/Violent/30.mp4\n",
      "2024-06-05 12:26:30,264 - INFO - Extracted frames from ./Data/Video/Violent/31.mp4\n",
      "2024-06-05 12:26:30,964 - INFO - Extracted frames from ./Data/Video/Violent/108.mp4\n",
      "2024-06-05 12:26:35,700 - INFO - Extracted frames from ./Data/Video/Violent/11.mp4\n",
      "2024-06-05 12:26:35,839 - INFO - Extracted frames from ./Data/Video/Violent/103.mp4\n",
      "2024-06-05 12:26:36,399 - INFO - Extracted frames from ./Data/Video/Violent/17.mp4\n",
      "2024-06-05 12:26:37,732 - INFO - Extracted frames from ./Data/Video/Violent/36.mp4\n",
      "2024-06-05 12:26:37,751 - INFO - Extracted frames from ./Data/Video/Violent/19.mp4\n",
      "2024-06-05 12:26:39,934 - INFO - Extracted frames from ./Data/Video/Violent/92.mp4\n",
      "2024-06-05 12:26:41,391 - INFO - Extracted frames from ./Data/Video/Violent/54.mp4\n",
      "2024-06-05 12:26:44,254 - INFO - Extracted frames from ./Data/Video/Violent/109.mp4\n",
      "2024-06-05 12:26:44,649 - INFO - Extracted frames from ./Data/Video/Violent/43.mp4\n",
      "2024-06-05 12:26:45,208 - INFO - Extracted frames from ./Data/Video/Violent/18.mp4\n",
      "2024-06-05 12:26:45,290 - INFO - Extracted frames from ./Data/Video/Violent/58.mp4\n",
      "2024-06-05 12:26:45,637 - INFO - Extracted frames from ./Data/Video/Violent/85.mp4\n",
      "2024-06-05 12:26:46,695 - INFO - Extracted frames from ./Data/Video/Violent/68.mp4\n",
      "2024-06-05 12:26:49,018 - INFO - Extracted frames from ./Data/Video/Violent/73.mp4\n",
      "2024-06-05 12:26:50,602 - INFO - Extracted frames from ./Data/Video/Violent/8.mp4\n",
      "2024-06-05 12:26:51,395 - INFO - Extracted frames from ./Data/Video/Violent/63.mp4\n",
      "2024-06-05 12:26:51,740 - INFO - Extracted frames from ./Data/Video/Violent/53.mp4\n",
      "2024-06-05 12:26:52,609 - INFO - Extracted frames from ./Data/Video/Violent/39.mp4\n",
      "2024-06-05 12:26:53,640 - INFO - Extracted frames from ./Data/Video/Violent/22.mp4\n",
      "2024-06-05 12:26:54,191 - INFO - Extracted frames from ./Data/Video/Violent/111.mp4\n",
      "2024-06-05 12:26:56,900 - INFO - Extracted frames from ./Data/Video/Violent/52.mp4\n",
      "2024-06-05 12:26:58,891 - INFO - Extracted frames from ./Data/Video/Violent/95.mp4\n",
      "2024-06-05 12:26:59,063 - INFO - Extracted frames from ./Data/Video/Violent/69.mp4\n",
      "2024-06-05 12:26:59,187 - INFO - Extracted frames from ./Data/Video/Violent/71.mp4\n",
      "2024-06-05 12:26:59,746 - INFO - Extracted frames from ./Data/Video/Violent/1.mp4\n",
      "2024-06-05 12:27:02,554 - INFO - Extracted frames from ./Data/Video/Violent/32.mp4\n",
      "2024-06-05 12:27:03,004 - INFO - Extracted frames from ./Data/Video/Violent/48.mp4\n",
      "2024-06-05 12:27:03,128 - INFO - Extracted frames from ./Data/Video/Violent/46.mp4\n",
      "2024-06-05 12:27:05,316 - INFO - Extracted frames from ./Data/Video/Violent/20.mp4\n",
      "2024-06-05 12:27:06,661 - INFO - Extracted frames from ./Data/Video/Violent/107.mp4\n",
      "2024-06-05 12:27:06,773 - INFO - Extracted frames from ./Data/Video/Violent/86.mp4\n",
      "2024-06-05 12:27:08,684 - INFO - Extracted frames from ./Data/Video/Violent/110.mp4\n",
      "2024-06-05 12:27:09,514 - INFO - Extracted frames from ./Data/Video/Violent/49.mp4\n",
      "2024-06-05 12:27:12,017 - INFO - Extracted frames from ./Data/Video/Violent/10.mp4\n",
      "2024-06-05 12:27:13,182 - INFO - Extracted frames from ./Data/Video/Violent/79.mp4\n",
      "2024-06-05 12:27:13,546 - INFO - Extracted frames from ./Data/Video/Violent/60.mp4\n",
      "2024-06-05 12:27:14,191 - INFO - Extracted frames from ./Data/Video/Violent/7.mp4\n",
      "2024-06-05 12:27:14,503 - INFO - Extracted frames from ./Data/Video/Violent/99.mp4\n",
      "2024-06-05 12:27:15,011 - INFO - Extracted frames from ./Data/Video/Violent/45.mp4\n",
      "2024-06-05 12:27:15,223 - INFO - Extracted frames from ./Data/Video/Violent/96.mp4\n",
      "2024-06-05 12:27:15,997 - INFO - Extracted frames from ./Data/Video/Violent/64.mp4\n",
      "2024-06-05 12:27:17,081 - INFO - Extracted frames from ./Data/Video/Violent/74.mp4\n",
      "2024-06-05 12:27:17,664 - INFO - Extracted frames from ./Data/Video/Violent/112.mp4\n",
      "2024-06-05 12:27:19,546 - INFO - Extracted frames from ./Data/Video/Violent/28.mp4\n",
      "2024-06-05 12:27:20,754 - INFO - Extracted frames from ./Data/Video/Violent/35.mp4\n",
      "2024-06-05 12:27:20,879 - INFO - Extracted frames from ./Data/Video/Violent/89.mp4\n",
      "2024-06-05 12:27:21,331 - INFO - Extracted frames from ./Data/Video/Violent/104.mp4\n",
      "2024-06-05 12:27:27,490 - INFO - Extracted frames from ./Data/Video/NonViolent/6.mp4\n",
      "2024-06-05 12:27:27,749 - INFO - Extracted frames from ./Data/Video/NonViolent/4.mp4\n",
      "2024-06-05 12:27:29,142 - INFO - Extracted frames from ./Data/Video/NonViolent/14.mp4\n",
      "2024-06-05 12:27:30,290 - INFO - Extracted frames from ./Data/Video/NonViolent/37.mp4\n",
      "2024-06-05 12:27:30,439 - INFO - Extracted frames from ./Data/Video/NonViolent/40.mp4\n",
      "2024-06-05 12:27:31,713 - INFO - Extracted frames from ./Data/Video/NonViolent/50.mp4\n",
      "2024-06-05 12:27:31,906 - INFO - Extracted frames from ./Data/Video/NonViolent/38.mp4\n",
      "2024-06-05 12:27:34,051 - INFO - Extracted frames from ./Data/Video/NonViolent/56.mp4\n",
      "2024-06-05 12:27:34,387 - INFO - Extracted frames from ./Data/Video/NonViolent/25.mp4\n",
      "2024-06-05 12:27:36,023 - INFO - Extracted frames from ./Data/Video/NonViolent/42.mp4\n",
      "2024-06-05 12:27:37,138 - INFO - Extracted frames from ./Data/Video/NonViolent/9.mp4\n",
      "2024-06-05 12:27:37,819 - INFO - Extracted frames from ./Data/Video/NonViolent/5.mp4\n",
      "2024-06-05 12:27:38,279 - INFO - Extracted frames from ./Data/Video/NonViolent/21.mp4\n",
      "2024-06-05 12:27:38,913 - INFO - Extracted frames from ./Data/Video/NonViolent/16.mp4\n",
      "2024-06-05 12:27:38,973 - INFO - Extracted frames from ./Data/Video/NonViolent/3.mp4\n",
      "2024-06-05 12:27:40,395 - INFO - Extracted frames from ./Data/Video/NonViolent/44.mp4\n",
      "2024-06-05 12:27:43,007 - INFO - Extracted frames from ./Data/Video/NonViolent/34.mp4\n",
      "2024-06-05 12:27:44,297 - INFO - Extracted frames from ./Data/Video/NonViolent/27.mp4\n",
      "2024-06-05 12:27:45,535 - INFO - Extracted frames from ./Data/Video/NonViolent/51.mp4\n",
      "2024-06-05 12:27:46,057 - INFO - Extracted frames from ./Data/Video/NonViolent/12.mp4\n",
      "2024-06-05 12:27:47,108 - INFO - Extracted frames from ./Data/Video/NonViolent/47.mp4\n",
      "2024-06-05 12:27:47,483 - INFO - Extracted frames from ./Data/Video/NonViolent/29.mp4\n",
      "2024-06-05 12:27:49,721 - INFO - Extracted frames from ./Data/Video/NonViolent/26.mp4\n",
      "2024-06-05 12:27:53,743 - INFO - Extracted frames from ./Data/Video/NonViolent/41.mp4\n",
      "2024-06-05 12:27:53,759 - INFO - Extracted frames from ./Data/Video/NonViolent/55.mp4\n",
      "2024-06-05 12:27:54,820 - INFO - Extracted frames from ./Data/Video/NonViolent/59.mp4\n",
      "2024-06-05 12:27:56,796 - INFO - Extracted frames from ./Data/Video/NonViolent/13.mp4\n",
      "2024-06-05 12:27:57,357 - INFO - Extracted frames from ./Data/Video/NonViolent/23.mp4\n",
      "2024-06-05 12:27:58,064 - INFO - Extracted frames from ./Data/Video/NonViolent/57.mp4\n",
      "2024-06-05 12:27:58,295 - INFO - Extracted frames from ./Data/Video/NonViolent/2.mp4\n",
      "2024-06-05 12:28:00,742 - INFO - Extracted frames from ./Data/Video/NonViolent/24.mp4\n",
      "2024-06-05 12:28:00,776 - INFO - Extracted frames from ./Data/Video/NonViolent/15.mp4\n",
      "2024-06-05 12:28:01,494 - INFO - Extracted frames from ./Data/Video/NonViolent/33.mp4\n",
      "2024-06-05 12:28:03,107 - INFO - Extracted frames from ./Data/Video/NonViolent/19.mp4\n",
      "2024-06-05 12:28:06,858 - INFO - Extracted frames from ./Data/Video/NonViolent/36.mp4\n",
      "2024-06-05 12:28:07,220 - INFO - Extracted frames from ./Data/Video/NonViolent/43.mp4\n",
      "2024-06-05 12:28:07,967 - INFO - Extracted frames from ./Data/Video/NonViolent/30.mp4\n",
      "2024-06-05 12:28:08,598 - INFO - Extracted frames from ./Data/Video/NonViolent/31.mp4\n",
      "2024-06-05 12:28:09,490 - INFO - Extracted frames from ./Data/Video/NonViolent/54.mp4\n",
      "2024-06-05 12:28:10,958 - INFO - Extracted frames from ./Data/Video/NonViolent/11.mp4\n",
      "2024-06-05 12:28:11,106 - INFO - Extracted frames from ./Data/Video/NonViolent/18.mp4\n",
      "2024-06-05 12:28:11,578 - INFO - Extracted frames from ./Data/Video/NonViolent/17.mp4\n",
      "2024-06-05 12:28:14,755 - INFO - Extracted frames from ./Data/Video/NonViolent/8.mp4\n",
      "2024-06-05 12:28:14,807 - INFO - Extracted frames from ./Data/Video/NonViolent/58.mp4\n",
      "2024-06-05 12:28:15,044 - INFO - Extracted frames from ./Data/Video/NonViolent/53.mp4\n",
      "2024-06-05 12:28:17,117 - INFO - Extracted frames from ./Data/Video/NonViolent/22.mp4\n",
      "2024-06-05 12:28:17,842 - INFO - Extracted frames from ./Data/Video/NonViolent/48.mp4\n",
      "2024-06-05 12:28:17,973 - INFO - Extracted frames from ./Data/Video/NonViolent/52.mp4\n",
      "2024-06-05 12:28:19,255 - INFO - Extracted frames from ./Data/Video/NonViolent/39.mp4\n",
      "2024-06-05 12:28:19,806 - INFO - Extracted frames from ./Data/Video/NonViolent/1.mp4\n",
      "2024-06-05 12:28:23,082 - INFO - Extracted frames from ./Data/Video/NonViolent/20.mp4\n",
      "2024-06-05 12:28:23,751 - INFO - Extracted frames from ./Data/Video/NonViolent/10.mp4\n",
      "2024-06-05 12:28:23,929 - INFO - Extracted frames from ./Data/Video/NonViolent/32.mp4\n",
      "2024-06-05 12:28:24,055 - INFO - Extracted frames from ./Data/Video/NonViolent/49.mp4\n",
      "2024-06-05 12:28:24,365 - INFO - Extracted frames from ./Data/Video/NonViolent/45.mp4\n",
      "2024-06-05 12:28:24,350 - INFO - Extracted frames from ./Data/Video/NonViolent/46.mp4\n",
      "2024-06-05 12:28:25,388 - INFO - Extracted frames from ./Data/Video/NonViolent/7.mp4\n",
      "2024-06-05 12:28:26,937 - INFO - Extracted frames from ./Data/Video/NonViolent/60.mp4\n",
      "2024-06-05 12:28:27,767 - INFO - Extracted frames from ./Data/Video/NonViolent/28.mp4\n",
      "2024-06-05 12:28:27,942 - INFO - Extracted frames from ./Data/Video/NonViolent/35.mp4\n",
      "2024-06-05 12:28:28,358 - INFO - Found 14411 files in ./Data/TempFrames/V\n",
      "2024-06-05 12:28:29,323 - INFO - Moved files from ./Data/TempFrames/V to ./Data/Frames/Training/V and ./Data/Frames/Validation/V\n",
      "2024-06-05 12:28:29,433 - INFO - Found 9711 files in ./Data/TempFrames/NV\n",
      "2024-06-05 12:28:30,074 - INFO - Moved files from ./Data/TempFrames/NV to ./Data/Frames/Training/NV and ./Data/Frames/Validation/NV\n",
      "2024-06-05 12:28:30,080 - INFO - Frames extracted and data split into training and validation directories successfully.\n"
     ]
    }
   ],
   "source": [
    "logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')\n",
    "\n",
    "# Define base directory\n",
    "base_dir = './Data'\n",
    "\n",
    "# Define video and directory paths\n",
    "video_dir = os.path.join(base_dir, 'Video')\n",
    "temp_frames_dir = os.path.join(base_dir, 'TempFrames')\n",
    "train_dir = os.path.join(base_dir, 'Frames', 'Training')\n",
    "validation_dir = os.path.join(base_dir, 'Frames', 'Validation')\n",
    "\n",
    "# Ensure directories exist\n",
    "for directory in [train_dir, validation_dir, temp_frames_dir]:\n",
    "    os.makedirs(os.path.join(directory, 'V'), exist_ok=True)\n",
    "    os.makedirs(os.path.join(directory, 'NV'), exist_ok=True)\n",
    "\n",
    "def extract_frames(video_path, output_dir, frame_rate=30):\n",
    "    \"\"\"\n",
    "    Extract frames from video at a specified rate (every 'frame_rate' frames).\n",
    "    \"\"\"\n",
    "    cap = cv2.VideoCapture(video_path)\n",
    "    count = 0\n",
    "    success = True\n",
    "    while success:\n",
    "        success, frame = cap.read()\n",
    "        if count % frame_rate == 0 and success:\n",
    "            frame_filename = os.path.join(output_dir, f\"{os.path.basename(video_path).split('.')[0]}_frame{count}.jpg\")\n",
    "            cv2.imwrite(frame_filename, frame)\n",
    "        count += 1\n",
    "    cap.release()\n",
    "    logging.info(f\"Extracted frames from {video_path}\")\n",
    "\n",
    "def process_videos(video_files, output_dir, frame_rate=30):\n",
    "    for video_file in video_files:\n",
    "        extract_frames(video_file, output_dir, frame_rate)\n",
    "\n",
    "# Function to split and move files\n",
    "def split_and_move_files(src_dir, train_dst_dir, val_dst_dir, test_size=0.2):\n",
    "    # List all files in the source directory\n",
    "    files = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]\n",
    "    \n",
    "    # Log the number of files found\n",
    "    logging.info(f\"Found {len(files)} files in {src_dir}\")\n",
    "    \n",
    "    if len(files) == 0:\n",
    "        raise ValueError(f\"No files found in directory: {src_dir}\")\n",
    "\n",
    "    # Split into training and validation\n",
    "    train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)\n",
    "    \n",
    "    # Move training files\n",
    "    for file in train_files:\n",
    "        os.rename(file, os.path.join(train_dst_dir, os.path.basename(file)))\n",
    "    \n",
    "    # Move validation files\n",
    "    for file in val_files:\n",
    "        os.rename(file, os.path.join(val_dst_dir, os.path.basename(file)))\n",
    "\n",
    "    logging.info(f\"Moved files from {src_dir} to {train_dst_dir} and {val_dst_dir}\")\n",
    "\n",
    "# Paths for violent and non-violent videos\n",
    "violent_src_dir = os.path.join(video_dir, 'Violent')\n",
    "non_violent_src_dir = os.path.join(video_dir, 'NonViolent')\n",
    "temp_violent_dir = os.path.join(temp_frames_dir, 'V')\n",
    "temp_non_violent_dir = os.path.join(temp_frames_dir, 'NV')\n",
    "train_violent_dir = os.path.join(train_dir, 'V')\n",
    "train_non_violent_dir = os.path.join(train_dir, 'NV')\n",
    "validation_violent_dir = os.path.join(validation_dir, 'V')\n",
    "validation_non_violent_dir = os.path.join(validation_dir, 'NV')\n",
    "\n",
    "# Get list of video files\n",
    "violent_files = [os.path.join(violent_src_dir, f) for f in os.listdir(violent_src_dir) if os.path.isfile(os.path.join(violent_src_dir, f))]\n",
    "non_violent_files = [os.path.join(non_violent_src_dir, f) for f in os.listdir(non_violent_src_dir) if os.path.isfile(os.path.join(non_violent_src_dir, f))]\n",
    "\n",
    "# Use multiprocessing to extract frames from videos\n",
    "def parallel_process_videos(video_files, output_dir, frame_rate=30):\n",
    "    pool = mp.Pool(mp.cpu_count())\n",
    "    for video_file in video_files:\n",
    "        pool.apply_async(extract_frames, args=(video_file, output_dir, frame_rate))\n",
    "    pool.close()\n",
    "    pool.join()\n",
    "\n",
    "# Process violent and non-violent videos in parallel\n",
    "parallel_process_videos(violent_files, temp_violent_dir)\n",
    "parallel_process_videos(non_violent_files, temp_non_violent_dir)\n",
    "\n",
    "# Split and move files for violent and non-violent frames\n",
    "split_and_move_files(temp_violent_dir, train_violent_dir, validation_violent_dir)\n",
    "split_and_move_files(temp_non_violent_dir, train_non_violent_dir, validation_non_violent_dir)\n",
    "\n",
    "logging.info(\"Frames extracted and data split into training and validation directories successfully.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Image data generators\n",
    "FRAME_SIZE = (150, 150)\n",
    "BATCH_SIZE = 20"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_datagen = ImageDataGenerator(\n",
    "    rescale=1./255,\n",
    "    rotation_range=40,\n",
    "    width_shift_range=0.2,\n",
    "    height_shift_range=0.2,\n",
    "    shear_range=0.2,\n",
    "    horizontal_flip=True,\n",
    "    fill_mode='nearest'\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "val_datagen = ImageDataGenerator(rescale=1./255)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 19296 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "# Flow training images in batches using train_datagen generator\n",
    "train_generator = train_datagen.flow_from_directory(\n",
    "    train_dir,\n",
    "    target_size=FRAME_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='binary',\n",
    "    shuffle=True\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found 4826 images belonging to 2 classes.\n"
     ]
    }
   ],
   "source": [
    "validation_generator = val_datagen.flow_from_directory(\n",
    "    validation_dir,\n",
    "    target_size=FRAME_SIZE,\n",
    "    batch_size=BATCH_SIZE,\n",
    "    class_mode='binary',\n",
    "    shuffle=True\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load pre-trained VGG19 model\n",
    "vgg19_base = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))\n",
    "\n",
    "# Freeze the base model\n",
    "vgg19_base.trainable = False\n",
    "\n",
    "# Add custom layers on top\n",
    "model = models.Sequential([\n",
    "    vgg19_base,\n",
    "    layers.Flatten(),\n",
    "    layers.Dense(512, activation='relu'),\n",
    "    layers.Dropout(0.5),\n",
    "    layers.Dense(1, activation='sigmoid')\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                    </span>┃<span style=\"font-weight: bold\"> Output Shape           </span>┃<span style=\"font-weight: bold\">       Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ vgg19 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Functional</span>)              │ ?                      │    <span style=\"color: #00af00; text-decoration-color: #00af00\">20,024,384</span> │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Flatten</span>)               │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                   │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dropout</span>)               │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                 │ ?                      │   <span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (unbuilt) │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                   \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape          \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m      Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩\n",
       "│ vgg19 (\u001b[38;5;33mFunctional\u001b[0m)              │ ?                      │    \u001b[38;5;34m20,024,384\u001b[0m │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ flatten (\u001b[38;5;33mFlatten\u001b[0m)               │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense (\u001b[38;5;33mDense\u001b[0m)                   │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dropout (\u001b[38;5;33mDropout\u001b[0m)               │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "├─────────────────────────────────┼────────────────────────┼───────────────┤\n",
       "│ dense_1 (\u001b[38;5;33mDense\u001b[0m)                 │ ?                      │   \u001b[38;5;34m0\u001b[0m (unbuilt) │\n",
       "└─────────────────────────────────┴────────────────────────┴───────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">20,024,384</span> (76.39 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m20,024,384\u001b[0m (76.39 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">20,024,384</span> (76.39 MB)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m20,024,384\u001b[0m (76.39 MB)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.compile(\n",
    "    loss='binary_crossentropy',\n",
    "    optimizer=tf.keras.optimizers.Adam(),\n",
    "    metrics=['accuracy']\n",
    ")\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/30\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-05 12:29:33.598638: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 16777216 exceeds 10% of free system memory.\n",
      "2024-06-05 12:29:33.676078: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 16777216 exceeds 10% of free system memory.\n",
      "2024-06-05 12:29:33.681970: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 16777216 exceeds 10% of free system memory.\n",
      "2024-06-05 12:29:33.734148: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 16777216 exceeds 10% of free system memory.\n",
      "2024-06-05 12:29:33.751165: W external/local_tsl/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 16777216 exceeds 10% of free system memory.\n",
      "/home/shobhits/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py:121: UserWarning: Your `PyDataset` class should call `super().__init__(**kwargs)` in its constructor. `**kwargs` can include `workers`, `use_multiprocessing`, `max_queue_size`. Do not pass these arguments to `fit()`, as they will be ignored.\n",
      "  self._warn_if_super_not_called()\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m 18/964\u001b[0m \u001b[37m━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[1m1:43:42\u001b[0m 7s/step - accuracy: 0.5079 - loss: 2.1658"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[15], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m history \u001b[38;5;241m=\u001b[39m \u001b[43mmodel\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m      2\u001b[0m \u001b[43m    \u001b[49m\u001b[43mtrain_generator\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      3\u001b[0m \u001b[43m    \u001b[49m\u001b[43msteps_per_epoch\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtrain_generator\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msamples\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m/\u001b[39;49m\u001b[38;5;241;43m/\u001b[39;49m\u001b[43m \u001b[49m\u001b[43mBATCH_SIZE\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      4\u001b[0m \u001b[43m    \u001b[49m\u001b[43mepochs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m30\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m      5\u001b[0m \u001b[43m    \u001b[49m\u001b[43mvalidation_data\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mvalidation_generator\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      6\u001b[0m \u001b[43m    \u001b[49m\u001b[43mvalidation_steps\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mvalidation_generator\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msamples\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m/\u001b[39;49m\u001b[38;5;241;43m/\u001b[39;49m\u001b[43m \u001b[49m\u001b[43mBATCH_SIZE\u001b[49m\n\u001b[1;32m      7\u001b[0m \u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/keras/src/utils/traceback_utils.py:117\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    115\u001b[0m filtered_tb \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    116\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 117\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfn\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    118\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m    119\u001b[0m     filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/keras/src/backend/tensorflow/trainer.py:314\u001b[0m, in \u001b[0;36mTensorFlowTrainer.fit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq)\u001b[0m\n\u001b[1;32m    312\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m step, iterator \u001b[38;5;129;01min\u001b[39;00m epoch_iterator\u001b[38;5;241m.\u001b[39menumerate_epoch():\n\u001b[1;32m    313\u001b[0m     callbacks\u001b[38;5;241m.\u001b[39mon_train_batch_begin(step)\n\u001b[0;32m--> 314\u001b[0m     logs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mtrain_function\u001b[49m\u001b[43m(\u001b[49m\u001b[43miterator\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    315\u001b[0m     logs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_pythonify_logs(logs)\n\u001b[1;32m    316\u001b[0m     callbacks\u001b[38;5;241m.\u001b[39mon_train_batch_end(step, logs)\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/util/traceback_utils.py:150\u001b[0m, in \u001b[0;36mfilter_traceback.<locals>.error_handler\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    148\u001b[0m filtered_tb \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m    149\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m--> 150\u001b[0m   \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfn\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwargs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    151\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m \u001b[38;5;167;01mException\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m    152\u001b[0m   filtered_tb \u001b[38;5;241m=\u001b[39m _process_traceback_frames(e\u001b[38;5;241m.\u001b[39m__traceback__)\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:833\u001b[0m, in \u001b[0;36mFunction.__call__\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    830\u001b[0m compiler \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mxla\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jit_compile \u001b[38;5;28;01melse\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnonXla\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    832\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m OptionalXlaContext(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_jit_compile):\n\u001b[0;32m--> 833\u001b[0m   result \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mkwds\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    835\u001b[0m new_tracing_count \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mexperimental_get_tracing_count()\n\u001b[1;32m    836\u001b[0m without_tracing \u001b[38;5;241m=\u001b[39m (tracing_count \u001b[38;5;241m==\u001b[39m new_tracing_count)\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py:878\u001b[0m, in \u001b[0;36mFunction._call\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    875\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_lock\u001b[38;5;241m.\u001b[39mrelease()\n\u001b[1;32m    876\u001b[0m \u001b[38;5;66;03m# In this case we have not created variables on the first call. So we can\u001b[39;00m\n\u001b[1;32m    877\u001b[0m \u001b[38;5;66;03m# run the first trace but we should fail if variables are created.\u001b[39;00m\n\u001b[0;32m--> 878\u001b[0m results \u001b[38;5;241m=\u001b[39m \u001b[43mtracing_compilation\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcall_function\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    879\u001b[0m \u001b[43m    \u001b[49m\u001b[43margs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mkwds\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_variable_creation_config\u001b[49m\n\u001b[1;32m    880\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    881\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_created_variables:\n\u001b[1;32m    882\u001b[0m   \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCreating variables on a non-first call to a function\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    883\u001b[0m                    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m decorated with tf.function.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py:139\u001b[0m, in \u001b[0;36mcall_function\u001b[0;34m(args, kwargs, tracing_options)\u001b[0m\n\u001b[1;32m    137\u001b[0m bound_args \u001b[38;5;241m=\u001b[39m function\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39mbind(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n\u001b[1;32m    138\u001b[0m flat_inputs \u001b[38;5;241m=\u001b[39m function\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39munpack_inputs(bound_args)\n\u001b[0;32m--> 139\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mfunction\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_call_flat\u001b[49m\u001b[43m(\u001b[49m\u001b[43m  \u001b[49m\u001b[38;5;66;43;03m# pylint: disable=protected-access\u001b[39;49;00m\n\u001b[1;32m    140\u001b[0m \u001b[43m    \u001b[49m\u001b[43mflat_inputs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcaptured_inputs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mfunction\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcaptured_inputs\u001b[49m\n\u001b[1;32m    141\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/concrete_function.py:1322\u001b[0m, in \u001b[0;36mConcreteFunction._call_flat\u001b[0;34m(self, tensor_inputs, captured_inputs)\u001b[0m\n\u001b[1;32m   1318\u001b[0m possible_gradient_type \u001b[38;5;241m=\u001b[39m gradients_util\u001b[38;5;241m.\u001b[39mPossibleTapeGradientTypes(args)\n\u001b[1;32m   1319\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m (possible_gradient_type \u001b[38;5;241m==\u001b[39m gradients_util\u001b[38;5;241m.\u001b[39mPOSSIBLE_GRADIENT_TYPES_NONE\n\u001b[1;32m   1320\u001b[0m     \u001b[38;5;129;01mand\u001b[39;00m executing_eagerly):\n\u001b[1;32m   1321\u001b[0m   \u001b[38;5;66;03m# No tape is watching; skip to running the function.\u001b[39;00m\n\u001b[0;32m-> 1322\u001b[0m   \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_inference_function\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcall_preflattened\u001b[49m\u001b[43m(\u001b[49m\u001b[43margs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1323\u001b[0m forward_backward \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_select_forward_and_backward_functions(\n\u001b[1;32m   1324\u001b[0m     args,\n\u001b[1;32m   1325\u001b[0m     possible_gradient_type,\n\u001b[1;32m   1326\u001b[0m     executing_eagerly)\n\u001b[1;32m   1327\u001b[0m forward_function, args_with_tangents \u001b[38;5;241m=\u001b[39m forward_backward\u001b[38;5;241m.\u001b[39mforward()\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:216\u001b[0m, in \u001b[0;36mAtomicFunction.call_preflattened\u001b[0;34m(self, args)\u001b[0m\n\u001b[1;32m    214\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mcall_preflattened\u001b[39m(\u001b[38;5;28mself\u001b[39m, args: Sequence[core\u001b[38;5;241m.\u001b[39mTensor]) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Any:\n\u001b[1;32m    215\u001b[0m \u001b[38;5;250m  \u001b[39m\u001b[38;5;124;03m\"\"\"Calls with flattened tensor inputs and returns the structured output.\"\"\"\u001b[39;00m\n\u001b[0;32m--> 216\u001b[0m   flat_outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcall_flat\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43margs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    217\u001b[0m   \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mfunction_type\u001b[38;5;241m.\u001b[39mpack_output(flat_outputs)\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/polymorphic_function/atomic_function.py:251\u001b[0m, in \u001b[0;36mAtomicFunction.call_flat\u001b[0;34m(self, *args)\u001b[0m\n\u001b[1;32m    249\u001b[0m \u001b[38;5;28;01mwith\u001b[39;00m record\u001b[38;5;241m.\u001b[39mstop_recording():\n\u001b[1;32m    250\u001b[0m   \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_bound_context\u001b[38;5;241m.\u001b[39mexecuting_eagerly():\n\u001b[0;32m--> 251\u001b[0m     outputs \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_bound_context\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mcall_function\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    252\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mname\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    253\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mlist\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43margs\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    254\u001b[0m \u001b[43m        \u001b[49m\u001b[38;5;28;43mlen\u001b[39;49m\u001b[43m(\u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfunction_type\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mflat_outputs\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    255\u001b[0m \u001b[43m    \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    256\u001b[0m   \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    257\u001b[0m     outputs \u001b[38;5;241m=\u001b[39m make_call_op_in_graph(\n\u001b[1;32m    258\u001b[0m         \u001b[38;5;28mself\u001b[39m,\n\u001b[1;32m    259\u001b[0m         \u001b[38;5;28mlist\u001b[39m(args),\n\u001b[1;32m    260\u001b[0m         \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_bound_context\u001b[38;5;241m.\u001b[39mfunction_call_options\u001b[38;5;241m.\u001b[39mas_attrs(),\n\u001b[1;32m    261\u001b[0m     )\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/context.py:1500\u001b[0m, in \u001b[0;36mContext.call_function\u001b[0;34m(self, name, tensor_inputs, num_outputs)\u001b[0m\n\u001b[1;32m   1498\u001b[0m cancellation_context \u001b[38;5;241m=\u001b[39m cancellation\u001b[38;5;241m.\u001b[39mcontext()\n\u001b[1;32m   1499\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m cancellation_context \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m-> 1500\u001b[0m   outputs \u001b[38;5;241m=\u001b[39m \u001b[43mexecute\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mexecute\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m   1501\u001b[0m \u001b[43m      \u001b[49m\u001b[43mname\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mdecode\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mutf-8\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1502\u001b[0m \u001b[43m      \u001b[49m\u001b[43mnum_outputs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mnum_outputs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1503\u001b[0m \u001b[43m      \u001b[49m\u001b[43minputs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtensor_inputs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1504\u001b[0m \u001b[43m      \u001b[49m\u001b[43mattrs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mattrs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1505\u001b[0m \u001b[43m      \u001b[49m\u001b[43mctx\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m   1506\u001b[0m \u001b[43m  \u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1507\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m   1508\u001b[0m   outputs \u001b[38;5;241m=\u001b[39m execute\u001b[38;5;241m.\u001b[39mexecute_with_cancellation(\n\u001b[1;32m   1509\u001b[0m       name\u001b[38;5;241m.\u001b[39mdecode(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mutf-8\u001b[39m\u001b[38;5;124m\"\u001b[39m),\n\u001b[1;32m   1510\u001b[0m       num_outputs\u001b[38;5;241m=\u001b[39mnum_outputs,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1514\u001b[0m       cancellation_manager\u001b[38;5;241m=\u001b[39mcancellation_context,\n\u001b[1;32m   1515\u001b[0m   )\n",
      "File \u001b[0;32m~/Documents/Mini Project/CrimeDetection/env/lib/python3.12/site-packages/tensorflow/python/eager/execute.py:53\u001b[0m, in \u001b[0;36mquick_execute\u001b[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[1;32m     51\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m     52\u001b[0m   ctx\u001b[38;5;241m.\u001b[39mensure_initialized()\n\u001b[0;32m---> 53\u001b[0m   tensors \u001b[38;5;241m=\u001b[39m \u001b[43mpywrap_tfe\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mTFE_Py_Execute\u001b[49m\u001b[43m(\u001b[49m\u001b[43mctx\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_handle\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdevice_name\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mop_name\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     54\u001b[0m \u001b[43m                                      \u001b[49m\u001b[43minputs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mattrs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mnum_outputs\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     55\u001b[0m \u001b[38;5;28;01mexcept\u001b[39;00m core\u001b[38;5;241m.\u001b[39m_NotOkStatusException \u001b[38;5;28;01mas\u001b[39;00m e:\n\u001b[1;32m     56\u001b[0m   \u001b[38;5;28;01mif\u001b[39;00m name \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "history = model.fit(\n",
    "    train_generator,\n",
    "    steps_per_epoch=train_generator.samples // BATCH_SIZE,\n",
    "    epochs=30,\n",
    "    validation_data=validation_generator,\n",
    "    validation_steps=validation_generator.samples // BATCH_SIZE\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "t = time.time()\n",
    "export_path_keras = \"./vgg19_{}.h5\".format(int(t))\n",
    "model.save(export_path_keras)\n",
    "print(f\"Model saved to {export_path_keras}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "acc = history.history['accuracy']\n",
    "val_acc = history.history['val_accuracy']\n",
    "loss = history.history['loss']\n",
    "val_loss = history.history['val_loss']\n",
    "\n",
    "epochs = range(len(acc))\n",
    "\n",
    "plt.figure(figsize=(8, 8))\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(epochs, acc, 'r', label='Training accuracy')\n",
    "plt.plot(epochs, val_acc, 'b', label='Validation accuracy')\n",
    "plt.title('Training and Validation Accuracy')\n",
    "plt.legend()\n",
    "\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(epochs, loss, 'r', label='Training loss')\n",
    "plt.plot(epochs, val_loss, 'b', label='Validation loss')\n",
    "plt.title('Training and Validation Loss')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "env",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
```

## File: Website/backend/app.py
```python
from flask import Flask, Response, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

model = tf.keras.models.load_model('Models/CustomCNN.h5')

def preprocess_frame(frame):
    frame = cv2.resize(frame, (150, 150))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)

def generate_frames(video_source):
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        prediction = model.predict(preprocessed_frame)
        label = 'Suspicious' if prediction[0] > 0.9 else 'Peaceful'
        
        print(f"Prediction: {prediction[0]}, Label: {label}")

        socketio.emit('prediction', {'prediction': float(prediction[0]), 'label': label}, namespace='/')

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['file']
    file.save('uploaded_video.mp4')
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/uploaded_video_feed')
def uploaded_video_feed():
    return Response(generate_frames('uploaded_video.mp4'), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)
```

## File: Website/frontend/src/app/globals.css
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body{
  background-color: black;
}
```

## File: Website/frontend/src/app/layout.js
```javascript
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Crime Detection",
  description: "Generated by create next app",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
```

## File: Website/frontend/src/app/page.js
```javascript
import Benefits from "@/components/benefits/Benefits";
import Contact from "@/components/contact/Contact";
import CrimeDetector from "@/components/crimedetector/CrimeDetector";
import Features from "@/components/features/Features";
import Footer from "@/components/footer/Footer";
import Navbar from "@/components/navbar/Navbar";
import Topsection from "@/components/topsection/Topsection";


export default function Home() {
  
  return (
    <main className="">
      <div style={{backgroundImage: "url(/cctv1.jpeg)"}} className="min-h-screen bg-cover">
        <Navbar/>
        <Topsection/>
      </div>
      <Features/>
      <CrimeDetector/>
      <Benefits/>
      <Contact/>
      <Footer/>
    </main>
  )
}
```

## File: Website/frontend/src/components/benefits/Benefits.jsx
```javascript
import React from 'react'

const Benefits = () => {
  return (
    <section id='benefits' className='text-white py-20'>
     <h1 className='text-4xl font-semibold text-center pb-10'>BENEFITS</h1>
      <div className='w-9/12 mx-auto flex justify-between'>
        <div style={{backgroundImage: "url(/c2tv.png)"}} className='rounded-md bg-cover w-[28%] flex flex-col gap-2 justify-end h-[500px] px-2 pb-5'>
          <h1 className='text-2xl font-semibold text-center'>Enhanced Security</h1>
          <p className='text-center'>Improve overall security with proactive monitoring.</p>
        </div>
        <div style={{backgroundImage: "url(/ceff.png)"}} className='bg-cover rounded-md w-[28%] flex flex-col gap-2 justify-end h-[500px] px-2 pb-5'>
          <h1 className='text-2xl font-semibold text-center'>Cost Effective</h1>
          <p className='text-center'>Reduce costs by minimizing the need for physical security.</p>
        </div>
        <div style={{backgroundImage: "url(/quick.png)"}} className='bg-cover rounded-md w-[28%] flex flex-col gap-2 justify-end h-[500px] px-2 pb-5'>
          <h1 className='text-2xl font-semibold text-center'>Quick Response</h1>
          <p className='text-center'>Enable faster response times to potential threats.</p>
        </div>
      </div>
    </section>
  )
}

export default Benefits
```

## File: Website/frontend/src/components/contact/Contact.jsx
```javascript
import React from 'react'

const Contact = () => {
  return (
    <section id='contact' className="flex flex-col items-center py-20 text-white">
        <h1 className='text-4xl font-semibold text-center pb-10'>GET IN TOUCH</h1>
        <form className='bg-slate-800 flex flex-col justify-center gap-2 p-8 w-4/12'>
            <input type='name' className='w-full outline-none bg-transparent border-[1px] border-slate-700 p-1' placeholder='Name'></input>
            <input type='email' className='w-full outline-none bg-transparent border-[1px] border-slate-700 p-1' placeholder='Email'></input>
            <input className='w-full outline-none bg-transparent border-[1px] border-slate-700 p-1' placeholder='Phone'></input>
            <textarea rows={2} className='w-full outline-none bg-transparent border-[1px] border-slate-700 p-1' placeholder='Message'></textarea>
            <button type='submit' className='py-2 bg-stone-700'>Submit</button>
        </form>
        <div className='py-5 text-center text-lg font-light'>
            <h4>Email: shbhtsemwal@gmail.com</h4>
            <h4>Phone: +916396430288</h4>
        </div>
    </section>
  )
}

export default Contact
```

## File: Website/frontend/src/components/crimedetector/CrimeDetector.jsx
```javascript
"use client";
import React, { useRef, useEffect, useState } from 'react';
import io from 'socket.io-client';

const socket = io('http://localhost:5000');

const CrimeDetector = () => {
    const [prediction, setPrediction] = useState('');
    const [videoUploaded, setVideoUploaded] = useState(false);
    const [isPlaying, setIsPlaying] = useState(false);
    const [source, setSource] = useState('');
    const [cameras, setCameras] = useState([]);
    const [selectedCamera, setSelectedCamera] = useState('');
    const iframeRef = useRef(null);

    useEffect(() => {
        socket.on('prediction', (data) => {
            setPrediction(`Prediction: ${data.prediction}, Label: ${data.label}`);
        });

        navigator.mediaDevices.enumerateDevices().then(devices => {
            const videoDevices = devices.filter(device => device.kind === 'videoinput');
            setCameras(videoDevices);
            if (videoDevices.length > 0) {
                setSelectedCamera(videoDevices[0].deviceId);
            }
        });

        return () => {
            socket.off('prediction');
        };
    }, []);

    const handleVideoUpload = async (event) => {
        const file = event.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);

            await fetch('http://localhost:5000/upload', {
                method: 'POST',
                body: formData,
            });

            setVideoUploaded(true);
            setSource('uploaded');
            startVideo();
        }
    };

    const startVideo = () => {
        if (iframeRef.current) {
            if (source === 'live') {
                iframeRef.current.src = `http://localhost:5000/video_feed?cameraId=${selectedCamera}`;
            } else if (source === 'uploaded') {
                iframeRef.current.src = 'http://localhost:5000/uploaded_video_feed';
            }
            setIsPlaying(true);
        }
    };

    const pauseVideo = () => {
        if (iframeRef.current) {
            iframeRef.current.src = '';
            setIsPlaying(false);
        }
    };

    const handleStopVideo = () => {
        setVideoUploaded(false);
        setPrediction('');
        setSource('');
        if (iframeRef.current) {
            iframeRef.current.src = '';
        }
        setIsPlaying(false);
    };

    return (
        <section id='cd' className="flex flex-col items-center py-10 text-white">
            <h1 className='text-4xl font-semibold text-center pb-10'>Crime Detector</h1>
            <div className="flex items-center gap-4 py-5 bg-slate-800 px-5 rounded-md mb-5">
                {cameras.length > 0 && (
                    <div className="flex gap-2 items-center">
                        <label className="text-white text-xl font-medium">Select Camera: </label>
                        <select
                            className="px-4 py-2 bg-slate-500 rounded-md"
                            value={selectedCamera}
                            onChange={(e) => setSelectedCamera(e.target.value)}
                        >
                            {cameras.map((camera) => (
                                <option key={camera.deviceId} value={camera.deviceId}>
                                    {camera.label || `Camera ${camera.deviceId}`}
                                </option>
                            ))}
                        </select>
                    </div>
                )}
                <button 
                    onClick={() => { setSource('live'); startVideo(); }} 
                    className="bg-green-600 hover:bg-green-800 rounded-md px-10 py-2 "
                >
                    Start Live Feed
                </button>
                <h4 className='px-5'>OR</h4>
                <label className='px-10 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-700 cursor-pointer' htmlFor="uploadvdo">Upload Video</label>
                <input 
                    id='uploadvdo'
                    name='uploadvdo'
                    type="file" 
                    accept="video/*" 
                    onChange={handleVideoUpload} 
                    className="hidden"
                />
            </div>
            {(videoUploaded || source === 'live') && (
                <div className="w-full max-w-[680px] overflow-x-hidden p-4 rounded-lg shadow-md bg-slate-800">
                    <iframe 
                        ref={iframeRef}
                        width="100%" 
                        height="500"
                        allow="autoplay"
                        className="mb-4"
                    ></iframe>
                    <div className="flex justify-between">
                        <button 
                            onClick={startVideo} 
                            className={`px-4 py-2 flex gap-1 justify-center items-center rounded-md ${isPlaying ? 'bg-gray-500' : 'bg-blue-500 hover:bg-blue-700'} text-white`}
                            disabled={isPlaying}
                        >
                            Play 
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
                                <path d="M6 4.5L17.5 12L6 19.5Z" />
                            </svg>
                        </button>
                        <button 
                            onClick={pauseVideo} 
                            className={`px-4 py-2 rounded-md flex justify-center items-center gap-1 ${!isPlaying ? 'bg-gray-500' : 'bg-yellow-500 hover:bg-yellow-700'} text-white`}
                            disabled={!isPlaying}
                        >
                            Pause
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg">
                            <rect x="6" y="6" width="4" height="12" />
                            <rect x="14" y="6" width="4" height="12" />
                            </svg>

                        </button>
                        <button 
                            onClick={handleStopVideo} 
                            className="px-4 py-2 flex justify-center items-center gap-1 bg-red-500 text-white rounded-md hover:bg-red-700"
                        >
                            Cancel Video
                        </button>
                    </div>
                </div>
            )}
            {prediction && (
                <p className="mt-4 p-4 bg-slate-700 rounded-md shadow-md text-white font-medium">{prediction}</p>
            )}
        </section>
    );
};

export default CrimeDetector;
```

## File: Website/frontend/src/components/features/Features.jsx
```javascript
import Image from 'next/image'
import React from 'react'

const Features = () => {
  return (
    <section id='features' className='mx-auto w-9/12 text-white py-20'>
        <h1 className='text-4xl font-semibold text-center pb-10'>FEATURES</h1>
        <div className='flex flex-col gap-10 '>
            <div className='flex justify-between items-center'>
                <div className='flex justify-center w-[45%]'>
                    <Image width={1000} height={1000} src={'/cctv.jpeg'}></Image>
                </div>
                <div className='flex w-[45%] justify-end'>
                    <div className='w-10/12 flex flex-col gap-5'>
                        <h3 className='text-3xl font-medium text-right'>Real-time Monitoring</h3>
                        <p className='text-right'>This feature goes beyond simple motion detection. It utilizes advanced video analytics powered by machine learning algorithms to analyze live camera feeds and identify suspicious activities in real-time. </p>
                    </div>
                </div>
            </div>
            <div className='flex flex-row-reverse justify-between items-center'>
                <div className='flex justify-center w-[45%]'>
                    <Image width={1000} height={1000} src={'/vdo.png'}></Image>
                </div>
                <div className='w-[45%]'>
                    <div className='w-10/12 flex flex-col gap-5'>
                        <h3 className='text-3xl font-medium'>Recorded Video Analysis</h3>
                        <p className=''>This feature allows users to upload pre-recorded video footage for analysis. The system employs machine learning models to identify potential criminal activity within the uploaded video</p>
                    </div>
                </div>
            </div>
        </div>
    </section>
  )
}

export default Features
```

## File: Website/frontend/src/components/footer/Footer.jsx
```javascript
import Image from 'next/image'
import Link from 'next/link'
import React from 'react'

const Footer = () => {
  return (
    <footer className='bg-slate-800 py-8'>
        <div className='flex gap-5 w-full items-center justify-center'>
            <Link href={"#"}><Image className='w-[40px] h-[40px]' width={1000} height={1000} src={"/ig.png"}></Image></Link>
            <Link href={"#"}><Image className='w-[50px] h-[50px]' width={1000} height={1000} src={"/fb.png"}></Image></Link>
            <Link href={"#"}><Image className='w-[50px] h-[50px]' width={1000} height={1000} src={"/x.png"}></Image></Link>
            <Link href={"#"}><Image className='w-[40px] h-[40px]' width={1000} height={1000} src={"/discord.png"}></Image></Link>
        </div>
        <h3 className='text-center font-light text-white pt-5'>© 2024 Abhay Semwal All rights reserved.</h3>
    </footer>
  )
}

export default Footer
```

## File: Website/frontend/src/components/navbar/Navbar.jsx
```javascript
"use client"
import Link from 'next/link'
import React, { useEffect, useState } from 'react'

const Navbar = () => {
    const [navbg, setNavBg] = useState(false); // Initial state doesn't depend on window

    const changeNavBg = () => {
      window.scrollY >= 50 ? setNavBg(true) : setNavBg(false);
    }
  
    useEffect(() => {
      changeNavBg();
      window.addEventListener('scroll', changeNavBg);
      return () => window.removeEventListener('scroll', changeNavBg);
    }, [])
  

  return (
    <div onScroll={changeNavBg} className={`z-10 fixed flex justify-between items-center w-full py-3 px-20 text-white transition duration-300 ease-in-out ${navbg ? "bg-black" : "bg-gray-500/15"}`}>
      <div className='flex justify-between w-full'>
        <div className='w-[40%]'>
          <Link href="/" className='font-semibold text-2xl'>Crime Detection</Link>
        </div>
        <div className='flex items-center justify-between w-[60%]'>
          <Link href="/">Home</Link>
          <Link href="#features">Features</Link>
          <Link href="#cd">Crime Detector</Link>
          <Link href="#benefits">Benefits</Link>
          <Link href="#contact">Contact Us</Link>
        </div>
      </div>
    </div>
  )
}

export default Navbar
```

## File: Website/frontend/src/components/topsection/Topsection.jsx
```javascript
import Link from 'next/link'
import React from 'react'

const Topsection = () => {
  return (
    <section className='text-white'>
        <div className='flex flex-col items-center gap-5 pt-28'>
            <h1 className='text-5xl font-semibold'>Advanced Crime Detection Through CCTV</h1>
            <h3 className='text-3xl font-light'>Leveraging AI to Enhance Security</h3>
            <div className='absolute bottom-14'>
            <Link href={"#cd"} className='bg-gray-700 text-lg rounded-md py-2 px-6 border-[1px] border-white font-semibold'>Get Started</Link>
            </div>
        </div>
    </section>
  )
}

export default Topsection
```

## File: Website/frontend/jsconfig.json
```json
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

## File: Website/frontend/next.config.mjs
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {};

export default nextConfig;
```

## File: Website/frontend/package.json
```json
{
  "name": "frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "axios": "^1.7.2",
    "next": "14.2.3",
    "react": "^18",
    "react-dom": "^18",
    "socket.io-client": "^4.7.5"
  },
  "devDependencies": {
    "eslint": "^8",
    "eslint-config-next": "14.2.3",
    "postcss": "^8",
    "tailwindcss": "^3.4.1"
  }
}
```

## File: Website/frontend/postcss.config.mjs
```javascript
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
  },
};

export default config;
```

## File: Website/frontend/README.md
```markdown
cd Website 

For frontend
i) cd frontend
ii)npm i
iii)npm run dev

For backend
i)cd backend
ii)pip3 install -r requirements.txt
iii)python app.py
```

## File: Website/frontend/tailwind.config.js
```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
    },
  },
  plugins: [],
};
```

## File: Website/requirements.txt
```
flask
flask_cors
flask_socketio
tensorflow
numpy
opencv-python
cv2
```

## File: .gitignore
```
/env
/Machine Learning/A-Dataset-for-Automatic-Violence-Detection-in-Videos/
/Machine Learning/Data/
/Website/frontend/.next/
/Website/frontend/node_modules/
```
