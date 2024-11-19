import cv2
import customtkinter as ctk
from threading import Thread
from PIL import Image
import numpy as np

# Web Camera
cap = cv2.VideoCapture(1)

#cap = cv2.VideoCapture(r'C:\Users\Kevin\Downloads\Time-lapse of stars moving through the night sky.mp4')

# Get the frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("GUI")

        

        blank_image = Image.fromarray(np.zeros((frame_height, frame_width, 3), dtype=np.uint8))
        image = ctk.CTkImage(light_image=blank_image, size=(frame_width*2, frame_height))

        self.image_frame = ctk.CTkFrame(self, corner_radius=10, fg_color='#000000')
        self.image_frame.pack(padx=10, pady=5, anchor='center', fill='both')

        self.image_label_left = ctk.CTkLabel(self.image_frame, image=image, text='')
        self.image_label_left.pack(padx=5, pady=5, anchor='w', side='left')

        self.template_tracker = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        iamge = ctk.CTkImage(light_image=Image.fromarray(self.template_tracker), size=(frame_width*2, frame_height))

        self.image_label_right = ctk.CTkLabel(self.image_frame, image=iamge, text='')
        self.image_label_right.pack(padx=5, pady=5, anchor='e', side='right')

        self.control_frame = ctk.CTkFrame(self, corner_radius=10)
        self.control_frame.pack(padx=10, pady=5, anchor='center', fill='both')

        self.filter_bttn = ctk.CTkButton(self.control_frame, text='Clear', corner_radius=10, command=self.clear)
        self.filter_bttn.pack(padx=10, pady=10)

        self.bw_filter_label = ctk.CTkLabel(self.control_frame, text='')
        self.bw_filter_label.pack(padx=10, pady=5, anchor='center')

        self.bw_slider = ctk.CTkSlider(self.control_frame, corner_radius=10, from_=0, to=254, command=lambda e:self.update_slider_label(self.bw_slider, self.bw_filter_label))
        self.bw_slider.pack(padx=10, pady=5, fill='x', anchor='center')


        self.canny_filter_label = ctk.CTkLabel(self.control_frame, text='')
        self.canny_filter_label.pack(padx=10, pady=5, anchor='center')

        self.canny_slider = ctk.CTkSlider(self.control_frame, corner_radius=10, from_=0, to=254, command=lambda e:self.update_slider_label(self.canny_slider, self.canny_filter_label))
        self.canny_slider.pack(padx=10, pady=5, fill='x', anchor='center')

        Thread(target=self.run_process).start()

    def clear(self):
        self.template_tracker = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    def run_process(self):

        ret, frame = cap.read()
        if ret:
            frame = self.image_filter(frame)
            self.update_main_image(image=Image.fromarray(frame), frame=self.image_label_left)
            self.update_main_image(image=Image.fromarray(self.template_tracker), frame=self.image_label_right)
            self.after(10, self.run_process)
    
    def image_filter(self, image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, int(self.bw_slider.get()), 255, cv2.THRESH_BINARY)
        white_parts = cv2.bitwise_and(image, image, mask=mask)

        edged = cv2.Canny(white_parts, int(self.canny_slider.get()), 255) 
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 10 and h <= 10:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, text=f'x:{x} y:{y} w:{w} h:{h}', org=(x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 255, 0), thickness=1)
                cv2.circle(self.template_tracker, (x, y), 1, (0, 255, 0), 1)

        return image
    
    def update_slider_label(self, slider, label):
        label.configure(text=str(int(slider.get())))

    def update_main_image(self, image, frame):
        image = ctk.CTkImage(light_image=image, size=(frame_width, frame_height))
        frame.configure(image=image)

    def close(self):
        cap.release()
        self.destroy()


if __name__ == '__main__':
    app = App()
    app.mainloop()