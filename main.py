import cv2
import time
import os
from camera_utils import SimpleCamera
from simple_matcher import VisualMatcher

class RoomMemoryApp:
    def __init__(self):
        self.camera = SimpleCamera()
        self.matcher = VisualMatcher(method="orb")
        self.current_place = None
        self.confidence = 0
        
    def setup(self):
        """Initialize the application"""
        print("=" * 50)
        print("🧠 ROOM MEMORY - Visual Place Recognition")
        print("=" * 50)
        
        # Load existing maps
        self.matcher.load_map("maps")
        
        # Start camera
        if not self.camera.start():
            print("❌ Failed to start camera")
            return False
        
        return True
    
    def interactive_mode(self):
        """Interactive mode with live camera"""
        print("\n🎮 Interactive Mode")
        print("Commands:")
        print("  's' - Save current view")
        print("  'm' - Mark as new place")
        print("  'r' - Recognize current place")
        print("  'l' - List all known places")
        print("  'q' - Quit")
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                break
            
            # Display with overlay
            display_frame = frame.copy()
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
            
            # Add info text
            if self.current_place:
                cv2.putText(display_frame, f"📍 {self.current_place}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Confidence: {self.confidence:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No place recognized", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Room Memory", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.camera.save_frame()
                
            elif key == ord('m'):
                place_name = input("\nEnter place name: ").strip()
                if place_name:
                    self.camera.save_frame(folder="maps", name=f"{place_name}.jpg")
                    print(f"✅ Saved '{place_name}' to maps!")
                    self.matcher.load_map("maps")  # Reload maps
                
            elif key == ord('r'):
                place, confidence = self.matcher.recognize_place(frame)
                if place:
                    self.current_place = place
                    self.confidence = confidence
                    print(f"🔍 Recognized: {place} (confidence: {confidence:.2f})")
                else:
                    self.current_place = None
                    print("❓ Unknown place")
                    
            elif key == ord('l'):
                self.list_places()
                
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
    
    def list_places(self):
        """List all known places"""
        print("\n📋 Known Places:")
        print("-" * 30)
        
        if not os.path.exists("maps"):
            print("No maps folder found")
            return
        
        places = [f.split('.')[0] for f in os.listdir("maps") 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not places:
            print("No places saved yet")
            return
        
        for i, place in enumerate(places, 1):
            print(f"{i}. {place}")
        print("-" * 30)
    
    def auto_recognition_mode(self, interval=2):
        """Automatically recognize place every few seconds"""
        print(f"\n🤖 Auto-Recognition Mode (every {interval} seconds)")
        print("Press 'q' to stop")
        
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                break
            
            # Recognize
            place, confidence = self.matcher.recognize_place(frame)
            
            if place:
                if place != self.current_place or confidence > self.confidence + 0.1:
                    print(f"📍 {place} (confidence: {confidence:.2f})")
                    self.current_place = place
                    self.confidence = confidence
            else:
                if self.current_place is not None:
                    print("❓ Lost recognition")
                    self.current_place = None
            
            # Show preview
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Auto Recognition", display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(interval)
        
        cv2.destroyAllWindows()
    
    def run(self):
        """Main application loop"""
        if not self.setup():
            return
        
        while True:
            print("\n📱 Select Mode:")
            print("1. Interactive Mode")
            print("2. Auto-Recognition Mode")
            print("3. Map Builder (Capture new places)")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                self.interactive_mode()
            elif choice == '2':
                self.auto_recognition_mode()
            elif choice == '3':
                self.map_builder_mode()
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    def map_builder_mode(self):
        """Guide user to capture a new place"""
        print("\n🏗️ Map Builder Mode")
        print("Capture multiple views of a new place for better recognition")
        
        place_name = input("Enter place name: ").strip()
        if not place_name:
            print("❌ Name required")
            return
        
        print(f"\n📸 Capturing '{place_name}'")
        print("Move around and capture from different angles")
        print("Press 's' to capture, 'q' when done")
        
        capture_count = 0
        while True:
            frame = self.camera.get_frame()
            if frame is None:
                break
            
            display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(display_frame, f"Capturing: {place_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Captures: {capture_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Map Builder", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save with timestamp
                timestamp = time.strftime("%H%M%S")
                filename = f"{place_name}_{timestamp}_{capture_count:02d}.jpg"
                self.camera.save_frame(folder="maps", name=filename)
                capture_count += 1
                print(f"✅ Capture {capture_count} saved")
                
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print(f"\n✅ Saved {capture_count} views of '{place_name}'")
        self.matcher.load_map("maps")  # Reload maps

if __name__ == "__main__":
    app = RoomMemoryApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    finally:
        app.camera.release()