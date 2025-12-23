import cv2
import numpy as np
from collections import deque
import os

class ObjectDetectorTracker:
    def __init__(self):
        self.cap = None
        self.tracking_points = deque(maxlen=64)
        self.object_detected = False
        self.detection_mode = "color"  # "color" atau "cnn"
        
        # CNN Model setup
        self.cnn_net = None
        self.cnn_classes = []
        self.load_cnn_model()
        
    def load_cnn_model(self):
        """muat pretrained MobileNet-SSD model untuk CNN detection"""
        prototxt_path = "deploy.prototxt"
        model_path = "mobilenet_iter_73000.caffemodel"
        
        # cek modelnya
        if os.path.exists(prototxt_path) and os.path.exists(model_path):
            try:
                self.cnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("✓ CNN Model loaded successfully!")
                
                # MobileNet-SSD classes (21 classes)
                self.cnn_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                   "sofa", "train", "tvmonitor"]
            except Exception as e:
                print(f"✗ Error loading CNN model: {e}")
                self.cnn_net = None
        else:
            print("✗ CNN model files not found!")
            print(f"  Missing: {prototxt_path} atau {model_path}")
            print("  CNN detection akan disabled. Gunakan color detection mode.")
    
    def apply_filters(self, frame):
        """Penerapan filtering untuk preprocessing"""
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        return blurred
    
    def detect_edges(self, frame):
        """Edge detection menggunakan Canny"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    def detect_with_cnn(self, frame):
        """Object detection menggunakan CNN (MobileNet-SSD)"""
        if self.cnn_net is None:
            return None, None, None, 0, 0
        
        h, w = frame.shape[:2]
        
        # Prepare input untuk CNN
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
                                     0.007843, (300, 300), 127.5)
        
        self.cnn_net.setInput(blob)
        detections = self.cnn_net.forward()
        
        # Find detection with highest confidence
        max_confidence = 0
        best_detection = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5 and confidence > max_confidence:  # Threshold 50%
                max_confidence = confidence
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_detection = {
                    'class_id': idx,
                    'class_name': self.cnn_classes[idx] if idx < len(self.cnn_classes) else "unknown",
                    'confidence': confidence,
                    'box': box.astype("int")
                }
        
        if best_detection:
            (startX, startY, endX, endY) = best_detection['box']
            
            # Calculate center
            cx = (startX + endX) // 2
            cy = (startY + endY) // 2
            
            # Create pseudo-contour for visualization
            contour = np.array([
                [[startX, startY]],
                [[endX, startY]],
                [[endX, endY]],
                [[startX, endY]]
            ])
            
            area = (endX - startX) * (endY - startY)
            
            return (cx, cy), contour, best_detection['class_name'], area, best_detection['confidence']
        
        return None, None, None, 0, 0
    
    def detect_object_by_color(self, frame):
        """Deteksi objek berdasarkan warna (biru)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Range warna biru
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        return mask
    
    def find_contours_and_features(self, mask):
        """Menemukan kontur dan ekstraksi features"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            
            M = cv2.moments(largest_contour)
            
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                return (cx, cy), largest_contour, area, perimeter
        
        return None, None, 0, 0
    
    def track_object(self, center):
        """Object tracking - menyimpan trajektori"""
        if center is not None:
            self.tracking_points.appendleft(center)
            self.object_detected = True
        else:
            self.object_detected = False
    
    def draw_augmented_reality(self, frame, center, contour, area, label=None, confidence=None):
        """Augmented Reality - overlay informasi pada objek"""
        if center is not None and contour is not None:
            # Gambar bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Gambar centroid
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            
            # Text overlay berbeda untuk CNN vs Color detection
            if self.detection_mode == "cnn" and label:
                cv2.putText(frame, f"CNN: {label}", (x, y - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if confidence:
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                cv2.putText(frame, f"Object Detected", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Area: {int(area)}", (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Gambar tracking trail
            for i in range(1, len(self.tracking_points)):
                if self.tracking_points[i - 1] is None or self.tracking_points[i] is None:
                    continue
                
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, self.tracking_points[i - 1], 
                        self.tracking_points[i], (255, 0, 0), thickness)
        
        return frame
    
    def add_info_panel(self, frame):
        """Menambahkan panel informasi"""
        height, width = frame.shape[:2]
        
        # Panel semi-transparan
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Informasi
        cv2.putText(frame, "Computer Vision + CNN Project", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        mode_text = f"Mode: {'CNN Detection' if self.detection_mode == 'cnn' else 'Color Detection'}"
        mode_color = (0, 255, 255) if self.detection_mode == 'cnn' else (255, 0, 255)
        cv2.putText(frame, mode_text, (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)
        
        status_text = "Status: " + ("TRACKING" if self.object_detected else "SEARCHING")
        status_color = (0, 255, 0) if self.object_detected else (0, 0, 255)
        cv2.putText(frame, status_text, (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        cv2.putText(frame, "Press 'q':quit | 'e':edges", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'm':toggle mode", (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self):
        """Main loop untuk menjalankan sistem"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Tidak dapat membuka kamera")
            return
        
        show_edges = False
        
        print("\n" + "="*50)
        print("SISTEM COMPUTER VISION + CNN")
        print("="*50)
        print(f"CNN Model: {'LOADED ✓' if self.cnn_net else 'NOT LOADED ✗'}")
        print(f"Detection Mode: {self.detection_mode.upper()}")
        print("\nKontrol:")
        print("  'q' - Keluar")
        print("  'e' - Toggle edge detection")
        print("  'm' - Toggle detection mode (Color/CNN)")
        print("="*50 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Tidak dapat membaca frame")
                break
            
            frame = cv2.flip(frame, 1)
            
            # 1. Filtering
            filtered_frame = self.apply_filters(frame)
            
            # 2. Object Detection (berdasarkan mode)
            if self.detection_mode == "cnn" and self.cnn_net is not None:
                # CNN Detection
                center, contour, class_name, area, confidence = self.detect_with_cnn(filtered_frame)
                label = class_name
            else:
                # Color Detection
                mask = self.detect_object_by_color(filtered_frame)
                center, contour, area, perimeter = self.find_contours_and_features(mask)
                label = None
                confidence = None
            
            # 3. Object Tracking
            self.track_object(center)
            
            # 4. Augmented Reality Overlay
            result_frame = self.draw_augmented_reality(frame, center, contour, area, label, confidence)
            
            # 5. Add Info Panel
            result_frame = self.add_info_panel(result_frame)
            
            # Tampilkan hasil
            cv2.imshow("Computer Vision + CNN - Object Detection & Tracking", result_frame)
            
            # Toggle edge detection view
            if show_edges:
                edges = self.detect_edges(frame)
                cv2.imshow("Edge Detection (Canny)", edges)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('e'):
                show_edges = not show_edges
                if not show_edges:
                    cv2.destroyWindow("Edge Detection (Canny)")
            elif key == ord('m'):
                # Toggle mode
                if self.cnn_net is not None:
                    self.detection_mode = "cnn" if self.detection_mode == "color" else "color"
                    self.tracking_points.clear()  # Clear tracking trail saat ganti mode
                    print(f"\n→ Mode switched to: {self.detection_mode.upper()}")
                else:
                    print("\n✗ CNN model not loaded. Cannot switch to CNN mode.")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

def main():
    print("\n🚀 Starting Computer Vision System...")
    detector = ObjectDetectorTracker()
    detector.run()
    print("\n✓ Program terminated successfully.")

if __name__ == "__main__":
    main()