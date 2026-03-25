import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, List
from image_processor import ImageProcessor

class VisualMatcher:
    """Enhanced visual place recognition with ensemble matching"""
    
    def __init__(self, method="ensemble"):
        self.method = method
        self.map_images = {}
        self.map_features = {}
        self.processor = ImageProcessor()
        
        # Initialize ORB detector
        self.orb_detector = cv2.ORB_create(nfeatures=1000)
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Try to initialize SIFT (may not be available in all OpenCV builds)
        try:
            self.sift_detector = cv2.SIFT_create()
            self.sift_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            self.has_sift = True
        except:
            self.sift_detector = None
            self.sift_matcher = None
            self.has_sift = False
            print("⚠️ SIFT not available, using ORB only")
        
        # Set primary detector based on method
        if method == "orb":
            self.detector = self.orb_detector
            self.matcher = self.orb_matcher
        elif method == "sift" and self.has_sift:
            self.detector = self.sift_detector
            self.matcher = self.sift_matcher
        elif method == "ensemble":
            self.detector = self.orb_detector  # Primary
            self.matcher = self.orb_matcher
        else:
            # Fallback to ORB
            self.detector = self.orb_detector
            self.matcher = self.orb_matcher
    
    def load_map(self, map_folder="maps"):
        """Load saved place images with preprocessing"""
        self.map_images.clear()
        self.map_features.clear()
        
        if not os.path.exists(map_folder):
            print(f"⚠️ Map folder '{map_folder}' not found")
            return
        
        # Group images by base place name
        place_groups = {}
        for filename in os.listdir(map_folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Remove extension first
                name_no_ext = os.path.splitext(filename)[0]
                # Extract base name (remove timestamps and counters if they follow an underscore)
                # Format: "Name_Timestamp.jpg" or "Name.jpg"
                base_name = name_no_ext.split('_')[0]
                
                if base_name not in place_groups:
                    place_groups[base_name] = []
                place_groups[base_name].append(filename)
        
        # Load and process images
        for place_name, filenames in place_groups.items():
            all_descriptors = []
            
            for filename in filenames:
                path = os.path.join(map_folder, filename)
                img = cv2.imread(path)
                
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Preprocess image
                    processed = self.processor.preprocess(img_rgb, resize=True, normalize=True)
                    
                    # Store first image as reference (use clean name)
                    if place_name not in self.map_images:
                        self.map_images[place_name] = processed
                    
                    # Extract features from all images
                    if self.detector is not None:
                        gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
                        if descriptors is not None:
                            all_descriptors.append(descriptors)
            
            # Combine descriptors from all images of this place
            if all_descriptors:
                self.map_features[place_name] = np.vstack(all_descriptors)
        
        print(f"🗺️ Loaded {len(self.map_images)} places from maps")
    
    def compare_histograms(self, img1, img2):
        """Simple color histogram comparison"""
        # Convert to HSV for better color representation
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_RGB2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # Normalize
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        # Compare
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(similarity, 0)  # Clip negative values to 0
    
    def match_orb(self, query_descriptors, map_name):
        """Match using ORB features"""
        if map_name not in self.map_features:
            return 0
        
        map_descriptors = self.map_features[map_name]
        if query_descriptors is None or map_descriptors is None:
            return 0
        
        try:
            matches = self.matcher.match(query_descriptors, map_descriptors)
            # More matches = more similar
            match_score = len(matches) / min(len(query_descriptors), len(map_descriptors))
            return min(match_score, 1.0)
        except:
            return 0
    
    def match_sift(self, query_descriptors, map_name: str) -> float:
        """Match using SIFT features"""
        if not self.has_sift or map_name not in self.map_features:
            return 0
        
        map_descriptors = self.map_features[map_name]
        if query_descriptors is None or map_descriptors is None:
            return 0
        
        try:
            matches = self.sift_matcher.match(query_descriptors, map_descriptors)
            # Calculate score based on match quality
            if len(matches) > 0:
                distances = [m.distance for m in matches]
                avg_distance = np.mean(distances)
                # Normalize score (lower distance is better)
                score = max(0, 1 - (avg_distance / 200))
                return min(score, 1.0)
            return 0
        except:
            return 0
    
    def ensemble_match(self, query_image: np.ndarray, map_name: str, map_image: np.ndarray) -> float:
        """Combine multiple matching methods for better accuracy"""
        scores = []
        weights = []
        
        # Preprocess query image
        processed_query = self.processor.preprocess(query_image, resize=True, normalize=True)
        query_gray = cv2.cvtColor(processed_query, cv2.COLOR_RGB2GRAY)
        
        # ORB matching
        orb_kp, orb_desc = self.orb_detector.detectAndCompute(query_gray, None)
        if orb_desc is not None:
            orb_score = self.match_orb(orb_desc, map_name)
            scores.append(orb_score)
            weights.append(0.4)  # 40% weight
        
        # SIFT matching (if available)
        if self.has_sift:
            sift_kp, sift_desc = self.sift_detector.detectAndCompute(query_gray, None)
            if sift_desc is not None:
                sift_score = self.match_sift(sift_desc, map_name)
                scores.append(sift_score)
                weights.append(0.4)  # 40% weight
        
        # Histogram matching
        hist_score = self.compare_histograms(processed_query, map_image)
        scores.append(hist_score)
        weights.append(0.2)  # 20% weight
        
        # Weighted average
        if scores:
            total_weight = sum(weights)
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return weighted_score
        
        return 0
    
    def recognize_place(self, query_image: np.ndarray, threshold: float = 0.3) -> Tuple[Optional[str], float]:
        """Recognize which place the query image is from"""
        if not self.map_images:
            return None, 0
        
        best_match = None
        best_score = 0
        
        for place_name, map_image in self.map_images.items():
            if self.method == "ensemble":
                score = self.ensemble_match(query_image, place_name, map_image)
            else:
                # Preprocess query
                processed_query = self.processor.preprocess(query_image, resize=True, normalize=True)
                query_gray = cv2.cvtColor(processed_query, cv2.COLOR_RGB2GRAY)
                query_kp, query_desc = self.detector.detectAndCompute(query_gray, None)
                
                # Try feature matching first
                if query_desc is not None:
                    score = self.match_orb(query_desc, place_name)
                else:
                    # Fallback to histogram comparison
                    score = self.compare_histograms(processed_query, map_image)
            
            if score > best_score:
                best_score = score
                best_match = place_name
        
        if best_score >= threshold:
            return best_match, best_score
        return None, best_score