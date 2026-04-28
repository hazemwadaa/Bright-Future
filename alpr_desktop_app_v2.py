import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import sys
import cv2
import pytesseract
from ultralytics import YOLO
import pandas as pd
from datetime import datetime

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ALPRDesktopApp:
    def __init__(self, root, model_path, log_file='vehicle_entry_log.csv'):
        self.root = root
        self.root.title("ALPR Desktop System")
        self.root.geometry("800x650")
        
        # Load Model
        try:
            self.detector = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Model Error", f"Could not load model: {e}")
            self.root.destroy()
            return
            
        self.log_file = log_file
        self.whitelist = ["ABC1234", "XYZ7890", "MH12DE1433", "DL3CAY9324"]
        
        # UI Setup
        self.setup_ui()
        
    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Automated License Plate Recognition", font=("Arial", 18, "bold"))
        header.pack(pady=10)
        
        # Image Display Area
        self.canvas = tk.Canvas(self.root, width=500, height=300, bg="#f0f0f0", highlightthickness=1, highlightbackground="gray")
        self.canvas.pack(pady=10)
        
        # Control Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(btn_frame, text="Upload Image", command=self.upload_image, width=20, height=2, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.view_log_btn = tk.Button(btn_frame, text="View Entry Log", command=self.view_log, width=20, height=2, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.view_log_btn.pack(side=tk.LEFT, padx=10)
        
        # Result Labels
        self.result_label = tk.Label(self.root, text="Plate Number: ---", font=("Arial", 14, "bold"))
        self.result_label.pack(pady=5)
        
        self.status_label = tk.Label(self.root, text="Status: ---", font=("Arial", 14))
        self.status_label.pack(pady=5)
        
        self.action_label = tk.Label(self.root, text="Action: ---", font=("Arial", 14), fg="blue")
        self.action_label.pack(pady=5)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
            
        # Process image
        self.display_image(file_path)
        self.process_alpr(file_path)
        
    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((500, 300))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(250, 150, image=self.tk_img)
        
    def preprocess_for_ocr(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def extract_text(self, plate_img):
        processed_img = self.preprocess_for_ocr(plate_img)
        # We assume tesseract is in the system path
        try:
            text = pytesseract.image_to_string(processed_img, config='--psm 7').strip()
            clean_text = ''.join(e for e in text if e.isalnum()).upper()
            return clean_text
        except Exception:
            return "OCR_ERROR"

    def log_entry(self, plate_number):
        status = "Authorized" if plate_number in self.whitelist else "Unauthorized"
        action = "Gate Opened" if status == "Authorized" else "Gate Closed"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log file in the same directory as the executable
        log_path = os.path.join(os.path.dirname(sys.argv[0]), self.log_file)
        
        if not os.path.exists(log_path):
            df = pd.DataFrame(columns=['Timestamp', 'Plate Number', 'Status', 'Action'])
            df.to_csv(log_path, index=False)
            
        new_entry = pd.DataFrame([[timestamp, plate_number, status, action]], 
                                columns=['Timestamp', 'Plate Number', 'Status', 'Action'])
        new_entry.to_csv(log_path, mode='a', header=False, index=False)
        return status, action

    def process_alpr(self, img_path):
        results = self.detector(img_path)
        img = cv2.imread(img_path)
        
        detected_plate = "Not Detected"
        status = "---"
        action = "---"
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size == 0: continue
                    
                plate_text = self.extract_text(plate_crop)
                if plate_text:
                    detected_plate = plate_text
                    status, action = self.log_entry(plate_text)
                    break
            if detected_plate != "Not Detected": break
            
        self.result_label.config(text=f"Plate Number: {detected_plate}")
        self.status_label.config(text=f"Status: {status}", fg="green" if status == "Authorized" else "red")
        self.action_label.config(text=f"Action: {action}")
        
    def view_log(self):
        log_path = os.path.join(os.path.dirname(sys.argv[0]), self.log_file)
        if os.path.exists(log_path):
            if os.name == 'nt':
                os.startfile(log_path)
            else:
                os.system(f"open {log_path}")
        else:
            messagebox.showinfo("Log Error", "Log file does not exist yet.")

if __name__ == "__main__":
    # Path to the model, bundled as data in PyInstaller
    MODEL_FILENAME = 'best.pt'
    MODEL_PATH = resource_path(MODEL_FILENAME)
    
    # If the model is not found in the bundle, look in the current directory
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(os.path.dirname(sys.argv[0]), MODEL_FILENAME)

    root = tk.Tk()
    app = ALPRDesktopApp(root, MODEL_PATH)
    root.mainloop()
