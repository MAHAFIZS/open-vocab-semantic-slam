\# Open-Vocabulary Semantic SLAM (Prototype)



This project implements a prototype pipeline that combines:



\- Open-vocabulary object detection (OWLv2)

\- Multi-frame temporal consistency filtering

\- Persistent object representation

\- Scene graph generation



The goal is to move from raw perception to \*\*structured semantic understanding\*\* of indoor environments.



\---



\## 🚀 Pipeline Overview

Video → Frame Extraction → Object Detection → Temporal Filtering → Scene Graph





\---



\## 🧠 Key Idea



Open-vocabulary detectors can produce semantically plausible but physically incorrect detections (e.g., reflections or noise).



This project demonstrates how \*\*temporal consistency across multiple frames\*\* can:



\- Remove one-off false detections

\- Group repeated detections into persistent objects

\- Produce a cleaner and more reliable scene representation



\---



\## 🔧 Features



\- Open-vocabulary detection (no fixed object classes)

\- Multi-frame object consistency

\- Simple data association via clustering

\- Persistent object tracking across frames

\- Scene graph with spatial relations:

&#x20; - `left\_of`

&#x20; - `above`

&#x20; - `near`



\---



\## 📂 Project Structure

open-vocab-semantic-slam/

│

├── detect.py # Single-image detection

├── detect\_video.py # Detection on video frames

├── extract\_frames.py # Video → frames

├── temporal\_scene\_graph.py # Multi-frame object grouping + scene graph

│

├── data/

│ └── test.jpg # Sample image

│

├── outputs/

│ ├── detections.jpg

│ └── temporal\_scene\_graph.png

│

├── requirements.txt

└── README.md





\---



\## ⚙️ Installation

pip install -r requirements.txt





\---



\## ▶️ Usage



\### 1. Extract frames from video

python extract\_frames.py





\### 2. Run detection on frames



\### 2. Run detection on frames

python detect\_video.py





\### 3. Build temporal scene graph



\---



\## 📊 Example Output



\### Detection

!\[Detection](outputs/detections.jpg)



\### Temporal Scene Graph

!\[Scene Graph](outputs/temporal\_scene\_graph.png)



\---



\## 🔬 Insights



\- Single-frame detection is noisy

\- Temporal consistency improves robustness

\- Naive clustering leads to duplicate objects

\- Data association is critical for semantic SLAM



\---



\## 🔮 Future Work



\- Integrate SLAM poses (RTAB-Map)

\- Use depth for true 3D object localization

\- Improve object tracking (data association)

\- Task-aware scene graphs for robotic manipulation



\---



\## 👨‍💻 Author



M A Hafiz  

Robotics \& AI Engineer (Simulation, Control \& Perception)

