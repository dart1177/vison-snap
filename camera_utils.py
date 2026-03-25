import cv2
import time
from datetime import datetime
import os

class SimpleCamera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
    def start(self):
        """Start camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise Exception(f"Cannot open camera {self.camera_id}")
        print(f"📷 Camera {self.camera_id} started")
        return True
    
    def get_frame(self):
        """Get current frame from camera"""
        if self.cap is None:
            self.start()
            
        ret, frame = self.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        return None
    
    def save_frame(self, folder="captures", name=None):
        """Save current frame to file"""
        frame = self.get_frame()
        if frame is None:
            return None
            
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"capture_{timestamp}.jpg"
            
        path = os.path.join(folder, name)
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"💾 Saved: {path}")
        return path
    
    def show_preview(self, window_name="Camera Preview"):
        """Show live camera preview with basic controls"""
        print("Press 's' to save, 'q' to quit")
        
        while True:
            frame = self.get_frame()
            if frame is None:
                break
                
            # Display frame
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, display_frame)
            
            # Key controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save frame
                self.save_frame()
                
            elif key == ord('m'):  # Mark as new place
                place_name = input("Enter place name: ")
                self.save_frame(folder="maps", name=f"{place_name}.jpg")
                print(f"📍 Saved '{place_name}' to maps!")
                
            elif key == ord('q'):  # Quit
                break
        
        self.release()
        cv2.destroyAllWindows()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            print("📷 Camera released")
    
    def __del__(self):
        self.release()