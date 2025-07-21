import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk
from io import BytesIO
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

class CardSelectorGUI:
    def __init__(self):
        self.root = None
        self.selected_printing = None
        self.current_index = 0
        self.printings = []
        self.photo_image = None
        self.photo_image_back = None
        self.window_x = None
        self.window_y = None
        
        # image caching and background loading
        self.image_cache: Dict[str, Image.Image] = {}
        self.loading_status: Dict[int, str] = {}  # index -> 'loading'/'loaded'/'failed'
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.loading_lock = threading.Lock()
        
    def request_scryfall_image(self, image_url: str) -> bytes:
        """fetch image from scryfall with proper rate limiting"""
        r = requests.get(image_url, headers={'user-agent': 'silhouette-card-maker/0.1', 'accept': '*/*'})
        r.raise_for_status()
        time.sleep(0.15)  # maintain api rate limits
        return r.content
        
    def load_image_to_cache(self, image_url: str, cache_key: str) -> Image.Image:
        """load image from url and cache it"""
        try:
            if cache_key in self.image_cache:
                return self.image_cache[cache_key]
                
            image_data = self.request_scryfall_image(image_url)
            image = Image.open(BytesIO(image_data))
            
            # cache the original image
            self.image_cache[cache_key] = image
            return image
            
        except Exception as e:
            print(f"error loading image {cache_key}: {e}")
            return None
            
    def background_load_printing(self, printing_index: int):
        """load a specific printing's images in background"""
        if printing_index >= len(self.printings):
            return
            
        printing = self.printings[printing_index]
        
        with self.loading_lock:
            if printing_index in self.loading_status:
                return  # already loading or loaded
            self.loading_status[printing_index] = 'loading'
        
        try:
            card_faces = printing.get('card_faces', [])
            image_uris = printing.get('image_uris', {})
            
            # determine what images we need to load
            images_to_load = []
            
            if len(card_faces) >= 2:
                # double-sided card
                front_face = card_faces[0]
                back_face = card_faces[1]
                
                front_image_uris = front_face.get('image_uris', {})
                back_image_uris = back_face.get('image_uris', {})
                
                if 'normal' in front_image_uris:
                    images_to_load.append(('front', front_image_uris['normal'], f"{printing_index}_front"))
                if 'normal' in back_image_uris:
                    images_to_load.append(('back', back_image_uris['normal'], f"{printing_index}_back"))
                    
            elif 'normal' in image_uris:
                # single-sided card with main image
                images_to_load.append(('single', image_uris['normal'], f"{printing_index}_single"))
            elif len(card_faces) >= 1 and 'image_uris' in card_faces[0]:
                # single-sided card with face image
                face_image_uris = card_faces[0]['image_uris']
                if 'normal' in face_image_uris:
                    images_to_load.append(('single', face_image_uris['normal'], f"{printing_index}_single"))
            
            # load all images for this printing
            for image_type, image_url, cache_key in images_to_load:
                self.load_image_to_cache(image_url, cache_key)
            
            with self.loading_lock:
                self.loading_status[printing_index] = 'loaded'
                
            # if this is the currently displayed printing, update the display
            if printing_index == self.current_index and self.root:
                self.root.after(0, self.update_display_from_cache)
                
        except Exception as e:
            print(f"error loading printing {printing_index}: {e}")
            with self.loading_lock:
                self.loading_status[printing_index] = 'failed'
                
    def start_background_loading(self):
        """start loading all printings in background, prioritizing first few"""
        if not self.printings:
            return
            
        # load first printing immediately (blocking)
        if len(self.printings) > 0:
            self.background_load_printing(0)
            
        # start background loading for remaining printings
        for i in range(1, min(len(self.printings), 10)):  # limit to first 10 for performance
            self.executor.submit(self.background_load_printing, i)
            
        # load remaining printings with lower priority
        if len(self.printings) > 10:
            def load_remaining():
                time.sleep(1)  # small delay to let first batch finish
                for i in range(10, len(self.printings)):
                    self.executor.submit(self.background_load_printing, i)
                    
            threading.Thread(target=load_remaining, daemon=True).start()
        
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
            
    def load_and_display_images(self, printing: dict):
        """load and display image(s) - handles both single and double-sided cards"""
        try:
            # try to display from cache first for instant feedback
            self.update_display_from_cache()
            
        except Exception as e:
            print(f"error loading images: {e}")
            self.display_no_image()
            
    def update_display_from_cache(self):
        """update display using cached images if available"""
        if not self.printings or self.current_index >= len(self.printings):
            return
            
        printing = self.printings[self.current_index]
        card_faces = printing.get('card_faces', [])
        image_uris = printing.get('image_uris', {})
        
        # check loading status
        with self.loading_lock:
            status = self.loading_status.get(self.current_index, 'not_started')
        
        if len(card_faces) >= 2:
            # double-sided card - try to get both faces from cache
            front_cache_key = f"{self.current_index}_front"
            back_cache_key = f"{self.current_index}_back"
            
            if front_cache_key in self.image_cache and back_cache_key in self.image_cache:
                self.display_double_sided_from_cache(card_faces, front_cache_key, back_cache_key)
            elif status == 'loading':
                self.display_loading_message("Loading double-sided card...")
            else:
                self.display_loading_message("Loading images...")
                
        else:
            # single-sided card
            cache_key = f"{self.current_index}_single"
            
            if cache_key in self.image_cache:
                self.display_single_from_cache(cache_key)
            elif status == 'loading':
                self.display_loading_message("Loading card image...")  
            else:
                self.display_loading_message("Loading image...")
                
    def display_loading_message(self, message: str):
        """show loading message while images load"""
        # clear any back image and hide back label
        self.photo_image = None
        self.photo_image_back = None
        
        if hasattr(self, 'image_label_back'):
            self.image_label_back.grid_remove()
        if hasattr(self, 'face_label_front'):
            self.face_label_front.grid_remove()
        if hasattr(self, 'face_label_back'):
            self.face_label_back.grid_remove()
            
        self.image_label.configure(text=message, image="")
            
    def display_single_from_cache(self, cache_key: str):
        """display single card from cache"""
        try:
            if cache_key not in self.image_cache:
                self.display_loading_message("Loading image...")
                return
                
            image = self.image_cache[cache_key].copy()
            image.thumbnail((300, 420), Image.Resampling.LANCZOS)
            
            self.photo_image = ImageTk.PhotoImage(image)
            
            # clear any back image and hide back label
            self.photo_image_back = None
            if hasattr(self, 'image_label_back'):
                self.image_label_back.grid_remove()
            if hasattr(self, 'face_label_front'):
                self.face_label_front.grid_remove()
            if hasattr(self, 'face_label_back'):
                self.face_label_back.grid_remove()
                
            self.image_label.configure(image=self.photo_image, text="")
            
        except Exception as e:
            print(f"error displaying cached image: {e}")
            self.display_no_image()
    
    def display_single_card(self, image_url: str):
        """display a single card image (legacy method - now loads to cache)"""
        try:
            cache_key = f"temp_{hash(image_url)}"
            image = self.load_image_to_cache(image_url, cache_key)
            if image:
                image_copy = image.copy()
                image_copy.thumbnail((300, 420), Image.Resampling.LANCZOS)
                
                self.photo_image = ImageTk.PhotoImage(image_copy)
                
                # clear any back image and hide back label
                self.photo_image_back = None
                if hasattr(self, 'image_label_back'):
                    self.image_label_back.grid_remove()
                if hasattr(self, 'face_label_front'):
                    self.face_label_front.grid_remove()
                if hasattr(self, 'face_label_back'):
                    self.face_label_back.grid_remove()
                    
                self.image_label.configure(image=self.photo_image, text="")
            else:
                self.display_no_image()
            
        except Exception as e:
            print(f"error loading single image: {e}")
            self.display_no_image()
            
    def display_double_sided_from_cache(self, card_faces: list, front_cache_key: str, back_cache_key: str):
        """display double-sided card from cache"""
        try:
            if front_cache_key not in self.image_cache or back_cache_key not in self.image_cache:
                self.display_loading_message("Loading double-sided card...")
                return
                
            front_face = card_faces[0]
            back_face = card_faces[1]
            
            # load front image from cache
            front_image = self.image_cache[front_cache_key].copy()
            front_image.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(front_image)
            
            # load back image from cache  
            back_image = self.image_cache[back_cache_key].copy()
            back_image.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image_back = ImageTk.PhotoImage(back_image)
            
            # update labels
            self.image_label.configure(image=self.photo_image, text="")
            
            # show back image (create if doesn't exist)
            if not hasattr(self, 'image_label_back'):
                self.create_back_image_elements()
            
            self.image_label_back.configure(image=self.photo_image_back, text="")
            self.image_label_back.grid()
            
            # show face names
            if hasattr(self, 'face_label_front'):
                self.face_label_front.configure(text=front_face.get('name', 'Front'))
                self.face_label_front.grid()
            if hasattr(self, 'face_label_back'):
                self.face_label_back.configure(text=back_face.get('name', 'Back'))
                self.face_label_back.grid()
                
        except Exception as e:
            print(f"error displaying cached double-sided card: {e}")
            self.display_no_image()
            
    def display_double_sided_card(self, card_faces: list):
        """display both sides of a double-sided card"""
        try:
            front_face = card_faces[0]
            back_face = card_faces[1]
            
            front_image_uris = front_face.get('image_uris', {})
            back_image_uris = back_face.get('image_uris', {})
            
            if 'normal' not in front_image_uris or 'normal' not in back_image_uris:
                self.display_no_image()
                return
                
            # load front image
            front_data = self.request_scryfall_image(front_image_uris['normal'])
            front_image = Image.open(BytesIO(front_data))
            front_image.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(front_image)
            
            # load back image
            back_data = self.request_scryfall_image(back_image_uris['normal'])
            back_image = Image.open(BytesIO(back_data))
            back_image.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image_back = ImageTk.PhotoImage(back_image)
            
            # update labels
            self.image_label.configure(image=self.photo_image, text="")
            
            # show back image (create if doesn't exist)
            if not hasattr(self, 'image_label_back'):
                self.create_back_image_elements()
            
            self.image_label_back.configure(image=self.photo_image_back, text="")
            self.image_label_back.grid()
            
            # show face names
            if hasattr(self, 'face_label_front'):
                self.face_label_front.configure(text=front_face.get('name', 'Front'))
                self.face_label_front.grid()
            if hasattr(self, 'face_label_back'):
                self.face_label_back.configure(text=back_face.get('name', 'Back'))
                self.face_label_back.grid()
            
        except Exception as e:
            print(f"error loading double-sided images: {e}")
            self.display_no_image()
            
    def display_no_image(self):
        """display no image available message"""
        self.photo_image = None
        self.photo_image_back = None
        
        # hide back elements if they exist
        if hasattr(self, 'image_label_back'):
            self.image_label_back.grid_remove()
        if hasattr(self, 'face_label_front'):
            self.face_label_front.grid_remove()
        if hasattr(self, 'face_label_back'):
            self.face_label_back.grid_remove()
            
        self.image_label.configure(text="no image available", image="")
        
    def create_back_image_elements(self):
        """create gui elements for back image display"""
        # use the image_frame as parent
        parent = self.image_frame
        
        # create face name labels
        self.face_label_front = ttk.Label(parent, text="Front", font=("Arial", 10, "bold"), anchor="center")
        self.face_label_front.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        self.face_label_back = ttk.Label(parent, text="Back", font=("Arial", 10, "bold"), anchor="center")
        self.face_label_back.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        
        # create back image label
        self.image_label_back = ttk.Label(parent, text="loading back image...", anchor="center")
        self.image_label_back.grid(row=2, column=1, padx=(5, 0), pady=(0, 10))
            
    def update_display(self):
        """update the display with current printing info"""
        if not self.printings:
            return
            
        printing = self.printings[self.current_index]
        
        # update counter label with loading status
        with self.loading_lock:
            loaded_count = sum(1 for status in self.loading_status.values() if status == 'loaded')
            
        counter_text = f"{self.current_index + 1} / {len(self.printings)}"
        if loaded_count < len(self.printings):
            counter_text += f" ({loaded_count}/{len(self.printings)} loaded)"
            
        self.counter_label.configure(text=counter_text)
        
        # update card info
        set_name = printing.get('set_name', 'Unknown Set')
        set_code = printing.get('set', '').upper()
        collector_number = printing.get('collector_number', '')
        rarity = printing.get('rarity', '').title()
        release_date = printing.get('released_at', '')
        
        # check if this is a double-sided card for additional info
        card_faces = printing.get('card_faces', [])
        is_double_sided = len(card_faces) >= 2
        
        info_text = f"Set: {set_name} ({set_code})\n"
        info_text += f"Collector #: {collector_number}\n"
        info_text += f"Rarity: {rarity}\n"
        info_text += f"Released: {release_date}"
        
        if is_double_sided:
            info_text += f"\n\nDouble-Sided Card:"
            if len(card_faces) >= 1:
                front_face = card_faces[0]
                info_text += f"\nFront: {front_face.get('name', 'Unknown')}"
                if 'mana_cost' in front_face:
                    info_text += f" - {front_face['mana_cost']}"
            if len(card_faces) >= 2:
                back_face = card_faces[1]  
                info_text += f"\nBack: {back_face.get('name', 'Unknown')}"
                if 'mana_cost' in back_face:
                    info_text += f" - {back_face['mana_cost']}"
        
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
        
        # load and display images - handle both regular and double-sided cards
        self.load_and_display_images(printing)
            
    def previous_card(self):
        """go to previous printing"""
        if self.printings and self.current_index > 0:
            self.current_index -= 1
            self.update_display()
            
            # trigger background loading for this printing if not loaded
            with self.loading_lock:
                if self.current_index not in self.loading_status:
                    self.executor.submit(self.background_load_printing, self.current_index)
            
    def next_card(self):
        """go to next printing"""
        if self.printings and self.current_index < len(self.printings) - 1:
            self.current_index += 1
            self.update_display()
            
            # trigger background loading for this printing if not loaded
            with self.loading_lock:
                if self.current_index not in self.loading_status:
                    self.executor.submit(self.background_load_printing, self.current_index)
            
    def select_card(self):
        """select current printing and close window"""
        if self.printings:
            self.selected_printing = self.printings[self.current_index]
        self.close_gui()
        
    def skip_card(self):
        """use first available printing (skip manual selection)"""
        if self.printings:
            self.selected_printing = self.printings[0]
            print(f"using first available printing for card")
        else:
            self.selected_printing = None
        self.close_gui()
        
    def close_gui(self):
        """properly close and cleanup gui"""
        if self.root:
            try:
                self.save_window_position()
                # clean up image references
                self.photo_image = None
                self.photo_image_back = None
                self.root.quit()
                self.root.destroy()
            except Exception as e:
                print(f"error during gui cleanup: {e}")
            finally:
                self.root = None
                
        # clean up background loading
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            print(f"error shutting down executor: {e}")
            
        # clear caches to free memory
        self.image_cache.clear()
        self.loading_status.clear()
        
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
                    try:
                        self.window_x = int(parts[1])
                        self.window_y = int(parts[2])
                    except ValueError:
                        # if parsing fails, just keep current position
                        pass
        
    def show_selection_dialog(self, card_name: str, printings: list) -> dict:
        """show gui for selecting card printing, returns selected printing or none"""
        try:
            self.printings = printings
            self.current_index = 0
            self.selected_printing = None
            
            # reset caches and loading state for new dialog
            self.image_cache.clear()
            self.loading_status.clear()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=3)
            
            # create main window - wider to accommodate double-sided cards
            self.root = tk.Tk()
            self.root.title(f"Select Card Art - {card_name}")
            self.root.geometry("750x750")
            self.root.resizable(False, False)
            
            # restore window position if we have one saved
            if self.window_x is not None and self.window_y is not None:
                self.root.geometry(f"750x750+{self.window_x}+{self.window_y}")
            
            # main frame
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # card name label
            name_label = ttk.Label(main_frame, text=card_name, font=("Arial", 16, "bold"))
            name_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
            
            # counter label
            self.counter_label = ttk.Label(main_frame, text="", font=("Arial", 12))
            self.counter_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
            
            # create image container frame to support side-by-side layout
            self.image_frame = ttk.Frame(main_frame)
            self.image_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10))
            self.image_frame.columnconfigure(0, weight=1)
            self.image_frame.columnconfigure(1, weight=1)
            
            # image display (front image goes in column 0)
            self.image_label = ttk.Label(self.image_frame, text="loading image...", anchor="center")
            self.image_label.grid(row=2, column=0, pady=(0, 10))
            
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
            
            skip_button = ttk.Button(button_frame, text="Use First Printing", command=self.skip_card)
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
            
            # initial display and start background loading
            if self.printings:
                print(f"starting background loading for {len(self.printings)} printings...")
                
                # start background loading - this will load first printing immediately
                self.start_background_loading()
                
                # update display (will show first printing when loaded)
                self.update_display()
                
                # start periodic refresh to update loading progress
                self.schedule_refresh()
                
            # show dialog and wait for user interaction
            self.root.mainloop()
            
            return self.selected_printing
            
        except tk.TclError as e:
            print(f"gui error (is display available?): {e}")
            print("falling back to first available printing")
            if self.root:
                self.close_gui()
            return printings[0] if printings else None
        except Exception as e:
            print(f"unexpected gui error: {e}")
            print("falling back to first available printing")
            if self.root:
                self.close_gui()
            return printings[0] if printings else None
            
    def schedule_refresh(self):
        """schedule periodic refresh of loading status"""
        if self.root:
            # refresh every 500ms while loading
            with self.loading_lock:
                loaded_count = sum(1 for status in self.loading_status.values() if status == 'loaded')
                
            if loaded_count < len(self.printings):
                self.update_display()  # refresh current display
                self.root.after(500, self.schedule_refresh)  # schedule next refresh 