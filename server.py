import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from simple_matcher import VisualMatcher
from database import PlaceDatabase
import uvicorn
import os
import shutil
import json
from typing import List, Optional
import qrcode
import io
import base64

app = FastAPI(title="VisionSnap")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Matcher and Database
matcher = VisualMatcher(method="ensemble")
db = PlaceDatabase()
MAPS_DIR = "maps"
os.makedirs(MAPS_DIR, exist_ok=True)

def sync_database():
    """Sync files in maps/ folder with database"""
    print("🔄 Syncing database with maps folder...")
    all_stats = db.get_all_places_stats()
    known_places_lower = [p['name'].lower() for p in all_stats]
    
    files_synced = 0
    if os.path.exists(MAPS_DIR):
        for filename in os.listdir(MAPS_DIR):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Get place name from filename
                name_no_ext = os.path.splitext(filename)[0]
                base_name = name_no_ext.split('_')[0]
                
                if base_name.lower() not in known_places_lower:
                    filepath = os.path.join(MAPS_DIR, filename)
                    db.add_place(base_name, filepath)
                    known_places_lower.append(base_name.lower())
                    files_synced += 1
                
    if files_synced > 0:
        print(f"✅ Synced {files_synced} new places from files to database")

# Load maps and sync
matcher.load_map(MAPS_DIR)
sync_database()

# Mount static files (Frontend)
os.makedirs("static", exist_ok=True)
try:
    app.mount("/static", StaticFiles(directory="static", html=True), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")
    print("Make sure 'static' folder exists with index.html")

def read_image_file(file_data) -> np.ndarray:
    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/recognize")
async def recognize_place(file: UploadFile = File(...)):
    contents = await file.read()
    img = read_image_file(contents)
    
    if img is None:
        print("❌ Error: Invalid image format received")
        return {"error": "Invalid image format"}
        
    place, confidence = matcher.recognize_place(img, threshold=0.15)
    
    # Record recognition in database
    if place and place != "Unknown":
        print(f"✅ Recognized: {place} (confidence: {confidence:.3f})")
        db.record_recognition(place, confidence)
    else:
        print(f"❓ Could not recognize place (best guess was {place} with {confidence:.3f})")
    
    return {
        "place": place if place else "Unknown",
        "confidence": float(confidence)
    }

@app.post("/learn")
async def learn_place(name: str = Form(...), file: UploadFile = File(...)):
    # Sanitize name for filename use
    safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
    if not safe_name:
        return {"success": False, "message": "Invalid place name"}

    print(f"📥 Received learn request for: {name} (safe: {safe_name})") 
    contents = await file.read()
    if not contents:
        return {"success": False, "message": "No image data received"}
        
    img = read_image_file(contents)
    if img is None:
        print(f"❌ Error: Invalid image format for learning {name}")
        return {"success": False, "message": "Invalid image format"}
    
    # Save to maps folder
    import time
    timestamp = int(time.time())
    filename = f"{safe_name}_{timestamp}.jpg"
    filepath = os.path.join(MAPS_DIR, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(contents)
        print(f"💾 Saved image to {filepath}")
        
        # Add to database
        success = db.add_place(name, filepath) # Use original name for DB
        if success:
            print(f"🗄️ Added {name} to database")
        else:
            print(f"⚠️ Failed to add {name} to database")
            return {"success": False, "message": "Database entry failed"}
            
        # Reload matcher to include new place
        matcher.load_map(MAPS_DIR)
        print(f"🔄 Matcher reloaded with {len(matcher.map_images)} places")
        
        return {"success": True, "message": f"Learned new place: {name}"}
    except Exception as e:
        print(f"❌ Error during learning: {e}")
        return {"success": False, "message": f"Server Error: {str(e)}"}

@app.get("/places")
def get_places():
    return {"places": list(matcher.map_images.keys())}

@app.delete("/places/{name}")
async def delete_place(name: str):
    """Delete a place and its associated data"""
    try:
        # Delete from database
        if not db.delete_place(name):
            raise HTTPException(status_code=404, detail="Place not found")
        
        # Delete image files
        deleted_files = []
        for filename in os.listdir(MAPS_DIR):
            if filename.startswith(name + "_") or filename == f"{name}.jpg":
                filepath = os.path.join(MAPS_DIR, filename)
                os.remove(filepath)
                deleted_files.append(filename)
        
        # Reload matcher
        matcher.load_map(MAPS_DIR)
        
        return {
            "success": True,
            "message": f"Deleted place: {name}",
            "files_deleted": deleted_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/places/{name}/stats")
async def get_place_stats(name: str):
    """Get statistics for a specific place"""
    stats = db.get_place_stats(name)
    if not stats:
        raise HTTPException(status_code=404, detail="Place not found")
    return stats

@app.get("/stats")
async def get_all_stats():
    """Get statistics for all places"""
    return {
        "places": db.get_all_places_stats(),
        "total_places": len(matcher.map_images),
        "recognition_history": db.get_recognition_history(limit=20)
    }

@app.get("/qrcode")
async def generate_qr_code():
    """Generate QR code for mobile access"""
    import socket
    try:
        hostname = socket.gethostname()
        ip_list = socket.gethostbyname_ex(hostname)[2]
        ip_addresses = [ip for ip in ip_list if not ip.startswith("127.")]
        
        if ip_addresses:
            url = f"http://{ip_addresses[0]}:8000/static/index.html"
        else:
            url = "http://localhost:8000/static/index.html"
        
        # Generate QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(url)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "qr_code": f"data:image/png;base64,{img_str}",
            "url": url
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/network-info")
def get_network_info():
    """Get local network IP addresses for mobile access"""
    import socket
    ip_addresses = []
    
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Get all IP addresses
        ip_list = socket.gethostbyname_ex(hostname)[2]
        
        # Filter out localhost
        ip_addresses = [ip for ip in ip_list if not ip.startswith("127.")]
        
    except Exception as e:
        print(f"Error getting network info: {e}")
    
    return {
        "ip_addresses": ip_addresses,
        "port": 8000,
        "urls": [f"http://{ip}:8000/static/index.html" for ip in ip_addresses]
    }

@app.post("/export")
async def export_data():
    """Export all places and statistics"""
    try:
        data = db.export_data()
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """Get current configuration"""
    config_path = "config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "threshold": 0.15,
        "matcher_method": "ensemble",
        "auto_recognition_interval": 3000
    }

@app.put("/config")
async def update_config(config: dict):
    """Update configuration"""
    try:
        config_path = "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return {"success": True, "config": config}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
