import tkinter as tk
from tkinter import filedialog
import cv2
import os
import shutil
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

class App:
    def __init__(self, master):
        self.master = master
        master.title("Bone Fracture Detection")

        # Create a frame to hold the buttons
        button_frame = tk.Frame(master, bg='black')
        button_frame.pack(side=tk.LEFT, padx=10)

        # Create a button to open an image file
        self.open_button = tk.Button(button_frame, text="Open Image", height= 1, width=10,
                                     command=self.open_image, padx=10, pady=10)
        self.open_button.pack(side=tk.TOP, pady=10)

        # Create a button to send the image to the model
        self.predict_button = tk.Button(button_frame, text="Predict Image", height= 1, width=10,
                                        command=self.predict_image, padx=10, pady=10)
        self.predict_button.pack(side=tk.TOP, pady=10)

        # Create a button to save the image
        self.save_button = tk.Button(button_frame, text="Save Image", height= 1, width=10,
                                     command=self.save_image, padx=10, pady=10)
        self.save_button.pack(side=tk.TOP, pady=10)

        # Create a button to clear the image from the canvas
        self.clear_button = tk.Button(button_frame, text="Clear Image", height= 1, width=10,
                                      command=self.clear_image, padx=10, pady=10)
        self.clear_button.pack(side=tk.TOP, pady=10)

        # Create a button to quit the application
        self.quit_button = tk.Button(button_frame, text="Quit", height= 1, width=10,
                                     command=self.master.destroy, padx=10, pady=10)
        self.quit_button.pack(side=tk.TOP, pady=10)

        # Create two canvases to display the image
        self.canvas1 = tk.Canvas(master, width=640, height=800, bg='black')
        self.canvas1.pack(side=tk.LEFT)

        self.canvas2 = tk.Canvas(master, width=640, height=800, bg='black')
        self.canvas2.pack(side=tk.LEFT)

        # Initialize image objects to None
        self.image1 = None
        self.image2 = None
        self.image_name = None
        
        # Initialize the model
        self.model = YOLO("model/best.pt")



    def open_image(self):
        self.delete_previous_preprocessed_images()
            
        # Open a file dialog to select an image file
        file_path = filedialog.askopenfilename()
        #ta
        self.image_name = file_path.split('/')[-1]
        # Open the image file using cv2.imread()
        img = cv2.imread(file_path)
        # Convert the image to RGB mode using cv2.cvtColor()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize the image using cv2.resize()
        img1 = cv2.resize(img, (640, 800))
        

        # Convert the images to Tkinter-compatible format
        self.image1 = ImageTk.PhotoImage(image=Image.fromarray(img1))
        self.image2 = self.roı_clahe_pre_process(img1)
        
        img_arr = np.asarray(self.image2)
        cv2.imwrite(os.path.join(os.getcwd(), "preprocessed Images", self.image_name) ,
                    img_arr)
        

        # Display the images on both canvases using cv2.imshow()
        self.canvas1.create_image(0, 0, image=self.image1, anchor=tk.NW)
       

        # Store the image file path
        self.image_path = file_path
      
        

    def predict_image(self):
        # Check if an image has been selected
        if self.image2 is None:
            return
        
        self.delete_previous_predictions()
        
            
        self.model.predict(os.path.join(os.getcwd(), 'preprocessed images', self.image_name),
                    imgsz=1024,
                    show=False,
                    save=True
                    )
        
        image_path = os.path.join(os.getcwd(), 'runs', 'detect', 'predict', self.image_name)
        self.predicted_image = np.asarray(cv2.imread(image_path))
        self.image2 = ImageTk.PhotoImage(image=Image.fromarray(self.predicted_image))
        self.canvas2.create_image(0, 0, image=self.image2, anchor=tk.NW)
        
        

    def save_image(self):
        # Check if an image has been selected
        if self.image2 is None:
            return

        # Save the image to a new location
        new_file_path = filedialog.askdirectory()
        print(new_file_path)
        cv2.imwrite(os.path.join(new_file_path, 'predicted_' + self.image_name),
                    self.predicted_image)



    def clear_image(self):
        # Delete any image objects from the canvases
        self.canvas1.delete("all")
        self.canvas2.delete("all")

        # Reset the image objects to None
        self.image1 = None
        self.image2 = None
        self.image_name = None
        self.predicted_image = None
        self.image_path = None
   
    
   
    def roı_clahe_pre_process(self , img):
      if len(img.shape) == 3: #Check if image is rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert rgb to gray scale
      
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #determine clahe values
      img = clahe.apply(img) #apply clahe transform on image
      return img 



    def delete_previous_predictions(self):
        if 'runs' in os.listdir(os.getcwd()):
            shutil.rmtree('runs')
       
            
       
    def delete_previous_preprocessed_images(self):
        for file in os.listdir('preprocessed Images'):
            os.remove(os.path.join(os.getcwd(), 'preprocessed Images', file))
            
            
            
# Create the Tkinter application
root = tk.Tk()
root.attributes('-fullscreen', True)
root.configure(bg="black")

app = App(root)

# Run the Tkinter event loop
root.mainloop()