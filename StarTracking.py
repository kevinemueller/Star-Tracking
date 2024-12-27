import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
from threading import Thread

cam = cv2.VideoCapture(0)
ret, frame = cam.read()
cam_h, cam_w, cam_c = frame.shape 

class Sliders(ctk.CTk):
    def __init__(self, parent_frame):
        super().__init__()

        self.r_label = ctk.CTkLabel(parent_frame, text="R: 0", width=50)
        self.r_label.grid(row=0, column=0, padx=(5, 0))
        self.r_slider = ctk.CTkSlider(parent_frame, from_=0, to=255, command=lambda e:self.update_label(slider=self.r_slider, label=self.r_label, base='R:'), width=1280)
        self.r_slider.grid(row=0, column=1, padx=5, pady=5)

        self.g_label = ctk.CTkLabel(parent_frame, text="G: 0", width=50)
        self.g_label.grid(row=1, column=0, padx=(5, 0))
        self.g_slider = ctk.CTkSlider(parent_frame, from_=0, to=255, command=lambda e:self.update_label(slider=self.g_slider, label=self.g_label, base='G:'), width=1280)
        self.g_slider.grid(row=1, column=1, padx=5, pady=(0,5))

        self.b_label = ctk.CTkLabel(parent_frame, text="B: 0", width=50)
        self.b_label.grid(row=2, column=0, padx=(5, 0))
        self.b_slider = ctk.CTkSlider(parent_frame, from_=0, to=255, command=lambda e:self.update_label(slider=self.b_slider, label=self.b_label, base='B:'), width=1280)
        self.b_slider.grid(row=2, column=1, padx=5, pady=(0,5))


        # Set Sliders to 0
        self.r_slider.set(0)
        self.g_slider.set(0)
        self.b_slider.set(0)

    def update_label(self, slider, label, base):
        label.configure(text=f"{base} {str(int(slider.get()))}")
        

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title('App')

        self.image_frame = ctk.CTkFrame(self, corner_radius=10)
        self.image_frame.pack(side='top', anchor='center', padx=10, pady=10, expand=True, fill='both')

        blank_image = np.zeros((cam_h, cam_w, cam_c), dtype=np.uint8)
        image = ctk.CTkImage(Image.fromarray(blank_image), size=(cam_w, cam_h))

        self.image_edges = ctk.CTkLabel(self.image_frame, image=image, text="")
        self.image_edges.grid(row=0, column=0, sticky='nesw')

        self.image_mask = ctk.CTkLabel(self.image_frame, image=image, text="")
        self.image_mask.grid(row=0, column=1, sticky='nesw')

        self.image_live = ctk.CTkLabel(self.image_frame, image=image, text="")
        self.image_live.grid(row=0, column=2, sticky='nesw')

        self.slider_frame = ctk.CTkFrame(self, corner_radius=10)
        self.slider_frame.pack(side='top', anchor='center', padx=10, pady=(0,10), expand=True, fill='x')

        self.sliders = Sliders(self.slider_frame)

        Thread(target=self.run).start()

    def run(self):

        while cam.isOpened():
            ret, frame = cam.read()

            if ret:
                self.trackers = self.add_trackers(image=frame)

                while self.trackers:
                    tracker = self.trackers.pop()
                    x, y, w, h = tracker['pos']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 191, 255), 1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = ctk.CTkImage(Image.fromarray(frame), size=(cam_w, cam_h))
            self.image_live.configure(image=image)
            self.image_live.image = image

    def get_contours(self, image):
        edges = cv2.Canny(image, 255, 255)
        edges = cv2.GaussianBlur(edges, (3, 3), 0)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        temp_frame = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
        image_mask = ctk.CTkImage(Image.fromarray(temp_frame), size=(cam_w, cam_h))
        self.image_edges.configure(image=image_mask)
        self.image_edges.image = image_mask

        return contours

    def add_trackers(self, image):
        
        trackers = []

        r = int(self.sliders.r_slider.get())
        g = int(self.sliders.g_slider.get())
        b = int(self.sliders.b_slider.get())

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([r, g, b])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        frame = cv2.bitwise_and(image, image, mask=mask)
        contours = self.get_contours(image=frame)

        max_size = 50
        min_size = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w <= max_size and h <= max_size and w > min_size and h > min_size:
                cropped_image = frame[y:y+h, x:x+w]
                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                mean = np.mean(gray)
                max_bright = {'max': mean, 'pos': (x, y, w, h)}
                trackers.append(max_bright)

        
        temp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_mask = ctk.CTkImage(Image.fromarray(temp_frame), size=(cam_w, cam_h))
        self.image_mask.configure(image=image_mask)
        self.image_mask.image = image_mask

        return trackers


if __name__ == '__main__':
    app = App()
    app.mainloop()

print('Ended')
