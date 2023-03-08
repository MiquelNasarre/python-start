from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageDraw, ImageOps
import os
import uuid

brush_size = 6
canvas_height = 500
canvas_width = 500
pixel_number = 16
directory = "./training_examples/numbers16/"

class PaintApp:
    def __init__(self, root, canvas_width, canvas_height, brush_size):
        self.root = root
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.brush_size = brush_size
        self.setup()

    def setup(self):
        # create canvas
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # create image object for drawing
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

        # bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        # create clear button
        self.clear_button = Button(self.root, text="Clear Canvas", command=self.clear_canvas)
        self.clear_button.pack()

        # create save button
        self.save_button = Button(self.root, text="Save Image", command=self.save_image)
        self.save_button.pack()

        # create label and entry for directory
        self.directory_label = Label(self.root, text="Directory:")
        self.directory_label.pack(side=LEFT)
        self.directory_entry = Entry(self.root, width=40)
        self.directory_entry.pack(side=LEFT)
        self.directory_entry.insert(END, directory)

        # create label for status
        self.status_label = Label(self.root, text="")
        self.status_label.pack()

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill='black')

    def reset(self, event):
        pass

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        self.draw = ImageDraw.Draw(self.image)

    def save_image(self):
        directory = self.directory_entry.get()

        # create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)

        # generate random filename for the text file
        filename = os.path.join(directory, f"{str(uuid.uuid4())[:8]}.txt")

        self.image = self.image.resize((pixel_number, pixel_number))
        self.image = ImageOps.grayscale(self.image)
        pixels = list(self.image.getdata())

        with open(filename, "w") as f:
            for i in range(0, len(pixels), pixel_number):
                row = pixels[i:i+pixel_number]
                row = [1 - int(p/255) for p in row]
                f.write(" ".join(str(p) for p in row) + "\n")

        self.status_label.config(text=f"Saved {filename}")

if __name__ == "__main__":
    root = Tk()
    root.title("Paint App")
    app = PaintApp(root, canvas_width=canvas_width, canvas_height=canvas_height, brush_size=brush_size)
    root.mainloop()