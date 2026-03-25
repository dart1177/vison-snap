# vison-snap
🧠 Real-time visual place &amp; object recognition system using OpenCV ensemble matching (ORB + SIFT + Histogram) with a FastAPI backend and live web UI.
# 🧠 VisionSnap — Visual Place Recognition System

A real-time visual recognition system that learns and identifies places/objects
using your webcam. Built with OpenCV, FastAPI, and a clean web interface 
accessible from any device on your local network.

## ✨ Features

- 🎯 **Ensemble Matching** — combines ORB + SIFT + Color Histogram for better accuracy
- 🎨 **Auto-Augmentation** — one training photo generates 12 variants automatically
- 📱 **Mobile Ready** — scan QR code to open on your phone
- 🌓 **Dark/Light Theme** — toggle between themes
- 📊 **Live Dashboard** — confidence trend, recognition history, stats
- 💾 **SQLite Database** — tracks recognition history and statistics
- 🔄 **Real-time** — auto-identifies every 3 seconds

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Vision | OpenCV (ORB + SIFT + FLANN), NumPy, Pillow |
| Frontend | HTML, CSS, Vanilla JS, Axios |
| Database | SQLite |
| Packaging | QRCode, Python-multipart |

## 🚀 Quick Start

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/VisionSnap.git
cd VisionSnap
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the server**
```bash
python server.py
```

**4. Open in browser**
```
http://localhost:8000
```

## 📖 How to Use

1. Click **Start** to activate your webcam
2. Click **Learn** → capture a photo → enter a name → **Save Location**
3. VisionSnap auto-recognizes every 3 seconds and shows match + confidence %
4. View history and stats in the dashboard

## 📁 Project Structure
```
VisionSnap/
├── server.py            # FastAPI backend + REST API
├── simple_matcher.py    # Core recognition engine (ORB + SIFT + Histogram)
├── image_processor.py   # Preprocessing + augmentation pipeline
├── camera_utils.py      # OpenCV camera wrapper
├── database.py          # SQLite place/history management
├── main.py              # Terminal mode (no web UI)
├── requirements.txt
└── static/
    ├── index.html       # Main web UI
    ├── app.js           # Frontend logic
    ├── style.css        # Styling + dark mode
    └── toast.js         # Notification system
```

## ⚙️ How Recognition Works

Each saved place stores three feature signatures:

- **ORB keypoints** — corners, edges, unique shapes (~2000 points)
- **SIFT keypoints** — scale & rotation invariant fine details
- **Color histogram** — HSV color distribution

At recognition time, the live camera frame is compared against all saved
places using a weighted ensemble:
```
Final Score = ORB (35%) + SIFT (45%) + Histogram (20%)
```

The place with the highest score above the threshold is returned as the match.

## 📸 Best Practices for Accuracy

- Capture **5-6 photos** per place from different angles
- Ensure **good lighting** when training
- For rooms: include distinctive landmarks (posters, furniture arrangements)
- For objects: fill the frame and avoid reflective surfaces

## 🔧 Known Limitations

- Works best for **rooms and large scenes** with many unique features
- Small shiny objects (watches, phones) are harder to distinguish
- Very dark / blank frames can cause false positives

## 🤝 Contributing

Pull requests welcome! Key areas for improvement:
- Deep learning model (MobileNet/EfficientNet) for better object recognition
- Multi-user support with authentication
- Cloud sync for place databases

## 📄 License

MIT License — free to use and modify.

---

Built with ❤️ using Python + OpenCV
```

---

## GitHub Topics to Add
Click the gear ⚙️ next to **About** on your repo and add these tags:
```
computer-vision opencv python fastapi place-recognition machine-learning 
real-time webcam image-processing sift orb
