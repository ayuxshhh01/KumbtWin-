# 👁️ KumbtWin - Smart Event Safety System

> **AI-Powered Real-Time Crowd Monitoring & Alert System for Large-Scale Events.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-orange.svg)
![Status](https://img.shields.io/badge/Status-Prototype-yellow.svg)

## 📖 Overview
**GuardianEye** is a comprehensive surveillance ecosystem designed to enhance safety at massive public gatherings (like the Kumbh Mela, concerts, or transport hubs). 

By integrating live CCTV/Webcam feeds with **YOLOv8 Computer Vision**, it autonomously counts crowd density in specific zones. If a zone exceeds its safety threshold, the system triggers instant alerts to security personnel via a real-time dashboard.

## ✨ Key Features
* **📹 Live Crowd Counting:** Uses YOLOv8 object detection to count people in real-time from video feeds.
* **⚡ Instant Alerts:** Broadcasts high-severity warnings via **WebSockets** when overcrowding is detected.
* **🛡️ Role-Based Access Control:** Secure login/signup with permissions for Admins, Staff, and Security Personnel.
* **📍 Zone Monitoring:** Maps detected individuals to specific geo-fenced areas (using Shapely).
* **📊 Live Dashboard:** View real-time statistics, active alerts, and system health.
* **🚀 Non-Blocking Performance:** AI processing runs on a background thread to keep the web server responsive.

## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **AI/ML:** Ultralytics YOLOv8, OpenCV
* **Database:** SQLite, SQLAlchemy (ORM)
* **Geospatial:** Shapely (Point-in-Polygon logic)
* **Frontend:** HTML5, CSS, JavaScript (Jinja2 Templates)
* **Authentication:** OAuth2 with JWT Tokens & BCrypt hashing

## 📂 Project Structure
```bash
KumbtWin/
├── main.py                 # The core application (API + AI Logic)
├── event_alert_system.db   # SQLite Database (Auto-generated)
├── templates/              # HTML Frontend files
│   ├── login_page.html
│   ├── main_dashboard.html
│   ├── user_management.html
│   └── ...
├── static/                 # CSS/JS/Images
└── README.md               # Documentation
