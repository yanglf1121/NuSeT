from tkinter import *
from tkinter.filedialog import askopenfilename, askdirectory, asksaveasfile
from tkinter.ttk import Progressbar
import PIL.Image, PIL.ImageTk
import numpy as np
import os
import tensorflow as tf
from test import test, test_single_img
from train_gui import train_and_score
TITLE_FONT = 'Arial 15 bold'
BUTTON_FONT = 'Arial 18'
PROGRESS_FONT = 'Arial 15'
BUTTON_FILL = None
BUTTON_WIDTH = 24
PADX = 3
PADY = 5
progressbarLen = 270
class NuSeT:
    def __init__(self, window, window_title):
        self.params = {}
        # Default values, most of them can be changed by user in the gui
        self.params['watershed'] = 'yes'
        self.params['min_score'] = 0.85
        self.params['nms_threshold'] = 0.1
        self.params['postProcess'] = 'yes'
        self.params['lr'] = 5e-5
        self.params['optimizer'] = 'rmsprop'
        self.params['epochs'] = 35
        self.params['normalization_method'] = 'fg'

        self.window = window
        self.window.title(window_title)
        self.window.geometry('250x410')

        self.window.configure()

        train_frame = Frame(self.window,highlightbackground="gray", highlightcolor="gray", highlightthickness=3, bd= 0)
        train_frame.pack(side="top")
        pred_frame = Frame(self.window,highlightbackground="gray", highlightcolor="gray", highlightthickness=3, bd= 0)
        pred_frame.pack(side="top")

        self.train_label = Label(train_frame, text="Training", font=TITLE_FONT)
        self.train_label.pack(side="top", fill='both', expand=True, padx=PADX, pady=PADY)
        
        self.training_configuration_btn = Button(train_frame, text="Configuration", 
        font=BUTTON_FONT, width=BUTTON_WIDTH, command=self.train_configuration)

        self.training_configuration_btn.pack(side="top", fill=BUTTON_FILL, expand=True, 
        padx=2, pady=2)

        self.train_btn = Button(train_frame, text="Begin training", font=BUTTON_FONT, 
        width=BUTTON_WIDTH, command=self.train)
        self.train_btn.pack(side="top", fill=BUTTON_FILL, expand=True, padx=PADX, pady=PADY)

        self.training_results = StringVar()
        self.train_progress_bar_label = Label(train_frame, text="Training Progress", 
        font=PROGRESS_FONT, width=BUTTON_WIDTH, foreground="gray", textvariable=self.training_results)

        self.train_progress_bar_label.pack(side="top", fill='x', expand=True, padx=PADX, pady=PADY)
        self.training_results.set('Training Progress')
        train_frame.update()

        self.train_progress_var = DoubleVar()
        self.train_progress = Progressbar(train_frame, orient="horizontal",
                                        length=progressbarLen, mode="determinate", 
                                        variable=self.train_progress_var, maximum=100)
        self.train_progress.pack(side="top", fill=BUTTON_FILL, expand=True, padx=PADX, pady=PADY)

        self.pred_label = Label(pred_frame, text="Predicting", font=TITLE_FONT)
        self.pred_label.pack(side="top", fill='both', expand=True, padx=PADX, pady=PADY)

        self.configuration_btn = Button(pred_frame, text="Configuration", 
        font=BUTTON_FONT, width=BUTTON_WIDTH, command=self.configuration)

        self.configuration_btn.pack(side="top", fill=BUTTON_FILL, expand=True, padx=PADX, pady=PADY)

        self.load_image_btn = Button(pred_frame, text="Load Image", font=BUTTON_FONT, 
        width=BUTTON_WIDTH, command=self.open_file)

        self.load_image_btn.pack(side="top", fill=BUTTON_FILL, expand=True, padx=PADX, pady=PADY)

        self.segment_btn = Button(pred_frame, text="Segment", font=BUTTON_FONT, 
        width=BUTTON_WIDTH, command=self.segmentation)

        self.segment_btn.pack(side="top", fill=BUTTON_FILL, expand=True, padx=PADX, pady=PADY)

        self.batch_segment_btn = Button(pred_frame, text="Batch Segment", 
        font=BUTTON_FONT, width=BUTTON_WIDTH, command=self.segmentation_batch)

        self.batch_segment_btn.pack(side="top", fill=BUTTON_FILL, expand=True, 
        padx=2, pady=2)

        self.progress_bar_label = Label(pred_frame, text="Segmentation Progress", 
        font=PROGRESS_FONT, foreground="gray", width=BUTTON_WIDTH)

        self.progress_bar_label.pack(side="top", fill='x', expand=True, padx=PADX, pady=PADY)
        self.progress_var = DoubleVar()
        self.progress = Progressbar(pred_frame, orient="horizontal",
                                        length=progressbarLen, mode="determinate", 
                                        variable=self.progress_var, maximum=100)

        self.progress.pack(side="top", fill=BUTTON_FILL, expand=True, padx=PADX, pady=PADY)


        self.window.mainloop()

    def display_image(self, im, sub_title):
        if sub_title == 'Image':
            win = Toplevel()
            win.title(sub_title)
            imgwidth = self.width
            imgheight = self.height
            canvas = Canvas(win,width=imgwidth,height=imgheight)
            img = PIL.ImageTk.PhotoImage(im)
            canvas.image = img
            canvas.pack()
            canvas.create_image(imgwidth/2,imgheight/2, image=img)
        else:
            win = Toplevel()
            win.title(sub_title)
            imgwidth = self.width
            imgheight = self.height
            canvas = Canvas(win,width=imgwidth,height=imgheight)
            img = PIL.ImageTk.PhotoImage(im)
            canvas.image = img
            canvas.pack()
            canvas.create_image(imgwidth/2,imgheight/2, image=img)
            self.save_img_btn = Button(win, text="Save", command=self.save_img)
            self.save_img_btn.pack(side="top", fill='both', expand=True, padx=4, pady=4)

    def open_file(self):
        self.image_path = askopenfilename(initialdir="C:/Users/",
                            filetypes =(("TIFF file", "*.tif"),("PNG file","*.png"),
                            ("JPEG file","*.jpg"),("All Files","*.*")),

                            title = "Choose a file."
                            )
        self.im = PIL.Image.open(self.image_path)
        self.im_np = np.asarray(self.im)
        self.height, self.width = self.im_np.shape[0], self.im_np.shape[1]
        self.display_image(self.im, sub_title="Image")

    def train(self):
        self.train_img_path = askdirectory(initialdir="C:/Users/",
                            title = "Choose a training image directory."
                            )
        self.train_img_path = self.train_img_path + '/'

        self.train_label_path = askdirectory(initialdir="C:/Users/",
                            title = "Choose a training label directory."
                            )
        self.train_label_path = self.train_label_path + '/'

        # Train with whole image norm for the first round
        self.params['normalization_method'] = 'wn'
        with tf.Graph().as_default():
            train_and_score(self)
        
        # Train with foreground normalization for the second round
        self.params['normalization_method'] = 'fg'
        with tf.Graph().as_default():
            train_and_score(self)
        

    def segmentation(self):
        if len(self.im_np.shape) == 3:
            if self.im_np.shape[2] == 3:
                # convert to grayscale first
                r, g, b = self.im_np[:,:,0], self.im_np[:,:,1], self.im_np[:,:,2]
                self.im_np = 0.2989 * r + 0.5870 * g + 0.1140 * b
            else:
                win = Toplevel()
                win.title("error dimension")
                error_label = Label(win, text="image is not grascale or RGB")
                error_label.pack(side="top", fill='both', expand=False, padx=1, pady=1)
                return
        self.fix_img_dimension()

        #self.display_image(PIL.Image.fromarray(self.im_np), sub_title="Grayscale")
        with tf.Graph().as_default():
            self.im_mask_np = test_single_img(self.params, [self.im_np])
            self.im_mask = PIL.Image.fromarray((self.im_mask_np*255))
            self.display_image(self.im_mask, sub_title="Segmentation Results")

    def segmentation_batch(self):
        self.batch_seg_path = askdirectory(initialdir="C:/Users/",
                            title = "Choose a segmentation directory."
                            )
        self.batch_seg_path = self.batch_seg_path + '/'
        with tf.Graph().as_default():
            test(self.params, self)

    def fix_img_dimension(self):
        self.height = self.height//16*16
        self.width = self.width//16*16
        self.im_np = self.im_np[:self.height, :self.width]
    
    def save_img(self):
        save_path = asksaveasfile(mode='w', defaultextension=".png", 
        filetypes=(("PNG file", "*.png"),("TIFF file", "*.tif"),("JPEG file", "*.jpg"),("All Files", "*.*") ))

        save_path = os.path.abspath(save_path.name)
        if save_path[-3:] == 'png' or save_path[-3:] == 'jpg':
            self.im_mask = self.im_mask.convert("L")
        self.im_mask.save(save_path)

    def configuration(self):
        win = Toplevel()
        win.title('configuration')
        win.geometry('250x180')
        frame1 = Frame(win)
        frame1.pack(side="top")
        frame2 = Frame(win)
        frame2.pack(side="top")
        frame3 = Frame(win)
        frame3.pack(side="top")
        frame4 = Frame(win)
        frame4.pack(side="top")
        frame5 = Frame(win)
        frame5.pack(side="top")

        self.watershed_option = IntVar(value=1)
        watershed_label = Label(frame1, text="Watershed")
        watershed_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.watershed_check_box = Checkbutton(frame1, text=" ", variable=self.watershed_option)
        self.watershed_check_box.pack(side="right", fill='both', expand=True, padx=5, pady=5)

        min_score_label = Label(frame2, text="Min detection score")
        min_score_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.min_score_text = Text(frame2, height=1, width=5, borderwidth=2, relief="groove")
        self.min_score_text.insert(END, "0.85")
        self.min_score_text.pack(side="right", fill='both', expand=True, padx=5, pady=5)

        NMS_label = Label(frame3, text="NMS overlapping ratio")
        NMS_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.NMS_text = Text(frame3, height=1, width=5, borderwidth=2, relief="groove")
        self.NMS_text.insert(END, "0.1")
        self.NMS_text.pack(side="right", fill='both', expand=True, padx=5, pady=5)
        
        self.postProcess_option = IntVar(value=1)
        postProcess_option_label = Label(frame4, text="Post-processing")
        postProcess_option_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.postProcess_check_box = Checkbutton(frame4, text=" ", variable=self.postProcess_option)
        self.postProcess_check_box.pack(side="right", fill='both', expand=True, padx=5, pady=5)

        self.save_configuration_btn = Button(frame5, text="Save", command=self.save_configurarion)
        self.save_configuration_btn.pack(side="top", fill='both', expand=True, padx=4, pady=4)

    def save_configurarion(self):
        if self.watershed_option.get() == 1:
            self.params['watershed'] = 'yes'
        else:
            self.params['watershed'] = 'no'

        if self.min_score_text.get("1.0","end-1c")=="" or \
        float(self.min_score_text.get("1.0","end-1c")) > 1 or \
        float(self.min_score_text.get("1.0","end-1c")) < 0:

            self.params['min_score'] = 0.85
        else:
            self.params['min_score'] = float(self.min_score_text.get("1.0","end-1c"))
        
        if self.NMS_text.get("1.0","end-1c")=="" or \
        float(self.NMS_text.get("1.0","end-1c")) > 1 or \
        float(self.NMS_text.get("1.0","end-1c")) < 0:

            self.params['nms_threshold'] = 0.1
        else:
            self.params['nms_threshold'] = float(self.NMS_text.get("1.0","end-1c"))

        self.params['nms_threshold'] = float(self.NMS_text.get("1.0","end-1c"))

        if self.postProcess_option.get() == 1:
            self.params['postProcess'] = 'yes'
        else:
            self.params['postProcess'] = 'no'
        
    def train_configuration(self):
        win = Toplevel()
        win.title('Configuration')
        win.geometry('250x150')
        frame1 = Frame(win)
        frame1.pack(side="top")
        frame2 = Frame(win)
        frame2.pack(side="top")
        frame3 = Frame(win)
        frame3.pack(side="top")
        frame4 = Frame(win)
        frame4.pack(side="top")

        learning_rate_label = Label(frame1, text="Learning rate")
        learning_rate_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.learning_rate_text = Text(frame1, height=1, width=8, borderwidth=2, relief="groove")
        self.learning_rate_text.insert(END, "0.0001")
        self.learning_rate_text.pack(side="right", fill='both', expand=True, padx=5, pady=5)

        epoch_label = Label(frame2, text="Number of epochs")
        epoch_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)
        self.epoch_text = Text(frame2, height=1, width=8, borderwidth=2, relief="groove")
        self.epoch_text.insert(END, "35")
        self.epoch_text.pack(side="right", fill='both', expand=True, padx=5, pady=5)

        self.optmizer = StringVar()
        self.rmsprop_radiobutton = Radiobutton(frame3, text='Rmsprop', variable=self.optmizer, value='rmsprop')
        self.adam_radiobutton = Radiobutton(frame3, text='Adam', variable=self.optmizer, value='adam')
        self.rmsprop_radiobutton.pack(side="right", fill='both', expand=True, padx=5, pady=5)
        self.adam_radiobutton.pack(side="right", fill='both', expand=True, padx=5, pady=5)
        opt_label = Label(frame3, text="Optimizer")
        opt_label.pack(side="left", fill='both', expand=True, padx=5, pady=5)

        self.save_configuration_btn = Button(frame4, text="Save", command=self.save_train_configurarion)
        self.save_configuration_btn.pack(side="top", fill='both', expand=True, padx=4, pady=4)

    def save_train_configurarion(self):

        if self.learning_rate_text.get("1.0","end-1c")=="" or \
        float(self.learning_rate_text.get("1.0","end-1c")) > 1 or \
        float(self.learning_rate_text.get("1.0","end-1c")) < 0:

            self.params['lr'] = 0.0001
        else:
            self.params['lr'] = float(self.learning_rate_text.get("1.0","end-1c"))
        
        self.params['epochs'] = int(self.epoch_text.get("1.0","end-1c"))
        self.params['optimizer'] = self.optmizer.get()

NuSeT(Tk(), "NuSeT")
