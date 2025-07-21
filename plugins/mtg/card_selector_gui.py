import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk
from io import BytesIO
import time
import sys

class CardSelectorGUI:
    def __init__(self):
        self.root = None
        self.selected_printing = None
        self.current_index = 0
        self.printings = []
        self.photo_image = None
        self.window_x = None
        self.window_y = None
        
    def request_scryfall_image(self, image_url: str) -> bytes:
        """fetch image from scryfall with proper rate limiting"""
        r = requests.get(image_url, headers={'user-agent': 'silhouette-card-maker/0.1', 'accept': '*/*'})
        r.raise_for_status()
        time.sleep(0.15)  # maintain api rate limits
        return r.content
        
    def load_and_display_image(self, image_url: str):
        """load image from url and display in gui"""
        try:
            image_data = self.request_scryfall_image(image_url)
            image = Image.open(BytesIO(image_data))
            
            # resize image to fit in gui (maintain aspect ratio)
            image.thumbnail((300, 420), Image.Resampling.LANCZOS)
            
            self.photo_image = ImageTk.PhotoImage(image)
            self.image_label.configure(image=self.photo_image)
            
        except Exception as e:
            print(f"error loading image: {e}")
            self.image_label.configure(text="image failed to load", image="")
            
    def update_display(self):
        """update the display with current printing info"""
        if not self.printings:
            return
            
        printing = self.printings[self.current_index]
        
        # update counter label
        counter_text = f"{self.current_index + 1} / {len(self.printings)}"
        self.counter_label.configure(text=counter_text)
        
        # update card info
        set_name = printing.get('set_name', 'Unknown Set')
        set_code = printing.get('set', '').upper()
        collector_number = printing.get('collector_number', '')
        rarity = printing.get('rarity', '').title()
        release_date = printing.get('released_at', '')
        
        info_text = f"Set: {set_name} ({set_code})\n"
        info_text += f"Collector #: {collector_number}\n"
        info_text += f"Rarity: {rarity}\n"
        info_text += f"Released: {release_date}"
        
        # add frame effects info if present
        frame_effects = printing.get('frame_effects', [])
        if frame_effects:
            info_text += f"\nFrame Effects: {', '.join(frame_effects)}"
            
        # add special treatments
        treatments = []
        if printing.get('full_art', False):
            treatments.append('Full Art')
        if printing.get('border_color') == 'borderless':
            treatments.append('Borderless')
        if printing.get('promo', False):
            treatments.append('Promo')
        if printing.get('digital', False):
            treatments.append('Digital')
            
        if treatments:
            info_text += f"\nTreatments: {', '.join(treatments)}"
            
        self.info_label.configure(text=info_text)
        
        # load and display image
        image_uris = printing.get('image_uris', {})
        if 'normal' in image_uris:
            self.load_and_display_image(image_uris['normal'])
        else:
            self.image_label.configure(text="no image available", image="")
            
    def previous_card(self):
        """go to previous printing"""
        if self.printings and self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            
    def next_card(self):
        """go to next printing"""
        if self.printings and self.current_index < len(self.printings) - 1:
            self.current_index += 1
            self.update_display()
            
    def select_card(self):
        """select current printing and close window"""
        if self.printings:
            self.selected_printing = self.printings[self.current_index]
        self.save_window_position()
        self.root.quit()
        self.root.destroy()
        
    def skip_card(self):
        """skip this card (don't download any version)"""
        self.selected_printing = None
        self.save_window_position()
        self.root.quit() 
        self.root.destroy()
        
    def save_window_position(self):
        """save current window position for next dialog"""
        if self.root:
            # get window position (x, y coordinates)
            self.root.update_idletasks()  # ensure geometry is updated
            geometry = self.root.geometry()
            # geometry format is "WIDTHxHEIGHT+X+Y"
            if '+' in geometry:
                parts = geometry.split('+')
                if len(parts) >= 3:
                    self.window_x = int(parts[1])
                    self.window_y = int(parts[2])
        
    def show_selection_dialog(self, card_name: str, printings: list) -> dict:
        """show gui for selecting card printing, returns selected printing or none"""
        try:
            self.printings = printings
            self.current_index = 0
            self.selected_printing = None
            
            # create main window
            self.root = tk.Tk()
            self.root.title(f"Select Card Art - {card_name}")
            self.root.geometry("500x700")
            self.root.resizable(False, False)
            
            # restore window position if we have one saved
            if self.window_x is not None and self.window_y is not None:
                self.root.geometry(f"500x700+{self.window_x}+{self.window_y}")
            
            # main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # card name label
            name_label = ttk.Label(main_frame, text=card_name, font=("Arial", 16, "bold"))
            name_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
            
            # counter label
            self.counter_label = ttk.Label(main_frame, text="", font=("Arial", 12))
            self.counter_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
            
            # image display
            self.image_label = ttk.Label(main_frame, text="loading image...", anchor="center")
            self.image_label.grid(row=2, column=0, columnspan=3, pady=(0, 10))
            
            # navigation buttons frame
            nav_frame = ttk.Frame(main_frame)
            nav_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
            
            prev_button = ttk.Button(nav_frame, text="◀ Previous", command=self.previous_card)
            prev_button.pack(side=tk.LEFT, padx=(0, 10))
            
            next_button = ttk.Button(nav_frame, text="Next ▶", command=self.next_card)
            next_button.pack(side=tk.LEFT)
            
            # card info label
            self.info_label = ttk.Label(main_frame, text="", justify=tk.LEFT, font=("Arial", 10))
            self.info_label.grid(row=4, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
            
            # action buttons frame
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
            
            select_button = ttk.Button(button_frame, text="Select This Version", command=self.select_card)
            select_button.pack(side=tk.LEFT, padx=(0, 10))
            
            skip_button = ttk.Button(button_frame, text="Skip Card", command=self.skip_card)
            skip_button.pack(side=tk.LEFT)
            
            # keyboard bindings
            self.root.bind('<Left>', lambda e: self.previous_card())
            self.root.bind('<Right>', lambda e: self.next_card())
            self.root.bind('<Return>', lambda e: self.select_card())
            self.root.bind('<Escape>', lambda e: self.skip_card())
            
            # configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=1)
            
            # initial display
            if self.printings:
                self.update_display()
                
            # show dialog and wait for user interaction
            self.root.mainloop()
            
            return self.selected_printing
            
        except tk.TclError as e:
            print(f"gui error (is display available?): {e}")
            print("falling back to first available printing")
            return printings[0] if printings else None
        except Exception as e:
            print(f"unexpected gui error: {e}")
            print("falling back to first available printing")
            return printings[0] if printings else None 