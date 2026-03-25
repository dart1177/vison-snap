# Quick test to verify camera works
from camera_utils import SimpleCamera

def quick_test():
    print("Testing camera...")
    cam = SimpleCamera()
    
    try:
        if cam.start():
            print("✅ Camera working!")
            
            # Take a test picture
            path = cam.save_frame("test_images", "test_capture.jpg")
            if path:
                print(f"✅ Test image saved to: {path}")
            
            # Show preview for 5 seconds
            print("Showing preview for 5 seconds...")
            import time
            
            start = time.time()
            while time.time() - start < 5:
                frame = cam.get_frame()
                if frame is not None:
                    # Show it
                    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Test", display)
                    cv2.waitKey(1)
            
            cv2.destroyAllWindows()
            print("✅ Test complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        cam.release()

if __name__ == "__main__":
    import cv2
    quick_test()