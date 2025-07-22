import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageDraw
from io import BytesIO
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional
import os
import pickle
import hashlib
import numpy as np

class CardSelectorGUI:
    # class-level persistent cache shared across all instances
    _global_image_cache: Dict[str, Image.Image] = {}
    _cache_lock = threading.Lock()
    _cache_dir = os.path.join(os.path.expanduser("~"), ".mtg_card_cache")
    _max_cache_size_mb = 500  # limit cache to 500mb
    _cache_access_times: Dict[str, float] = {}  # for lru eviction

    def __init__(self, use_caching=True, simple_cache=False):
        self.root = None
        self.selected_printing = None
        self.current_index = 0
        self.printings = []
        self.photo_image = None
        self.photo_image_back = None
        self.window_x = None
        self.window_y = None
        self.use_caching = use_caching
        self.simple_cache = simple_cache
        
        # per-dialog loading status (not cached globally)
        self.loading_status: Dict[int, str] = {}  # index -> 'loading'/'loaded'/'failed'
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.loading_lock = threading.Lock()
        
        # simple in-memory cache (just for this session)
        self.simple_image_cache: Dict[str, Image.Image] = {}
        
        # image adjustment parameters
        self.brightness = 1.0
        self.contrast = 1.0  
        self.saturation = 1.0
        self.gamma = 1.0
        self.color_balance = 0.0  # -100 to +100, negative=cooler, positive=warmer
        
        # store original images for adjustment (before processing)
        self.original_image = None
        self.original_image_back = None
        
        # selection system for targeted adjustments
        self.selection_mode = 'none'  # 'none', 'brush', 'color'
        self.selection_mask = None  # PIL Image mask (same size as card)
        self.selection_mask_back = None  # mask for back side
        self.brush_size = 10
        self.brush_mode = 'add'  # 'add' or 'subtract'
        self.current_draw_mode = 'add'  # mode for current drawing session
        self.color_tolerance = 30
        self.is_drawing = False
        self.last_draw_pos = None
        
        # selection overlay canvas
        self.selection_canvas = None
        self.selection_canvas_back = None
        
        # ensure cache directory exists only if advanced caching is enabled
        if self.use_caching:
            try:
                os.makedirs(self._cache_dir, exist_ok=True)
                print(f"cache directory: {self._cache_dir}")
            except Exception as e:
                print(f"warning: could not create cache directory: {e}")
                print("image caching will be disabled for this session")
                self.use_caching = False
        elif self.simple_cache:
            print("simple in-memory caching enabled")
        else:
            print("caching disabled - using simple mode")
        
    def request_scryfall_image(self, image_url: str) -> bytes:
        """fetch image from scryfall with proper rate limiting"""
        r = requests.get(image_url, headers={'user-agent': 'silhouette-card-maker/0.1', 'accept': '*/*'})
        r.raise_for_status()
        time.sleep(0.15)  # maintain api rate limits
        return r.content
        
    def _get_cache_key(self, image_url: str) -> str:
        """generate a cache key from image url"""
        return hashlib.md5(image_url.encode()).hexdigest()
    
    def _get_disk_cache_path(self, cache_key: str) -> str:
        """get the disk cache file path for a cache key"""
        return os.path.join(self._cache_dir, f"{cache_key}.pkl")
    
    def _load_from_disk_cache(self, cache_key: str) -> Image.Image:
        """load image from disk cache if available"""
        cache_path = self._get_disk_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"error loading disk cache {cache_key}: {e}")
                # remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass
        return None
    
    def _save_to_disk_cache(self, cache_key: str, image: Image.Image):
        """save image to disk cache"""
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(image, f)
        except Exception as e:
            print(f"error saving disk cache {cache_key}: {e}")
    
    def _estimate_cache_size_mb(self) -> float:
        """estimate current memory cache size in mb"""
        total_pixels = 0
        with self._cache_lock:
            for image in self._global_image_cache.values():
                total_pixels += image.width * image.height
        # rough estimate: 4 bytes per pixel (rgba) 
        return (total_pixels * 4) / (1024 * 1024)
    
    def _evict_lru_cache_entries(self):
        """evict least recently used entries if cache is too large"""
        current_size = self._estimate_cache_size_mb()
        if current_size <= self._max_cache_size_mb:
            return
            
        print(f"cache size {current_size:.1f}mb exceeds limit {self._max_cache_size_mb}mb, evicting entries...")
        
        with self._cache_lock:
            # sort by access time (oldest first)
            sorted_keys = sorted(self._cache_access_times.keys(), 
                               key=lambda k: self._cache_access_times[k])
            
            # remove oldest entries until we're under the limit
            for key in sorted_keys:
                if self._estimate_cache_size_mb() <= self._max_cache_size_mb * 0.8:  # 80% of limit
                    break
                    
                if key in self._global_image_cache:
                    del self._global_image_cache[key]
                if key in self._cache_access_times:
                    del self._cache_access_times[key]

    def load_image_to_cache(self, image_url: str, display_cache_key: str) -> Image.Image:
        """load image from url and cache it persistently"""
        try:
            url_cache_key = self._get_cache_key(image_url)
            current_time = time.time()
            
            # check memory cache first
            with self._cache_lock:
                if url_cache_key in self._global_image_cache:
                    self._cache_access_times[url_cache_key] = current_time
                    print(f"cache hit (memory): {display_cache_key}")
                    return self._global_image_cache[url_cache_key]
            
            # check disk cache
            image = self._load_from_disk_cache(url_cache_key)
            if image is not None:
                with self._cache_lock:
                    self._global_image_cache[url_cache_key] = image
                    self._cache_access_times[url_cache_key] = current_time
                print(f"cache hit (disk): {display_cache_key}")
                return image
            
            # download from internet
            print(f"cache miss, downloading: {display_cache_key}")
            image_data = self.request_scryfall_image(image_url)
            image = Image.open(BytesIO(image_data))
            
            # save to both memory and disk cache
            with self._cache_lock:
                self._global_image_cache[url_cache_key] = image
                self._cache_access_times[url_cache_key] = current_time
            
            # save to disk cache in background to avoid blocking
            def save_to_disk():
                self._save_to_disk_cache(url_cache_key, image)
            threading.Thread(target=save_to_disk, daemon=True).start()
            
            # check if we need to evict old entries
            self._evict_lru_cache_entries()
            
            return image
            
        except Exception as e:
            print(f"error loading image {display_cache_key}: {e}")
            return None
            
    @classmethod  
    def clear_all_cache(cls):
        """clear both memory and disk caches - useful for troubleshooting"""
        with cls._cache_lock:
            cls._global_image_cache.clear()
            cls._cache_access_times.clear()
            
        # clear disk cache
        try:
            import shutil
            if os.path.exists(cls._cache_dir):
                shutil.rmtree(cls._cache_dir)
                os.makedirs(cls._cache_dir, exist_ok=True)
                print("cleared all image caches")
        except Exception as e:
            print(f"error clearing disk cache: {e}")
            
    @classmethod
    def get_cache_stats(cls):
        """get cache statistics"""
        with cls._cache_lock:
            memory_count = len(cls._global_image_cache)
            memory_size_mb = 0
            if memory_count > 0:
                total_pixels = sum(img.width * img.height for img in cls._global_image_cache.values())
                memory_size_mb = (total_pixels * 4) / (1024 * 1024)
        
        disk_count = 0
        disk_size_mb = 0
        try:
            if os.path.exists(cls._cache_dir):
                cache_files = [f for f in os.listdir(cls._cache_dir) if f.endswith('.pkl')]
                disk_count = len(cache_files)
                
                disk_size_bytes = sum(
                    os.path.getsize(os.path.join(cls._cache_dir, f)) 
                    for f in cache_files
                )
                disk_size_mb = disk_size_bytes / (1024 * 1024)
        except Exception as e:
            print(f"error checking disk cache stats: {e}")
            
        return {
            'memory_images': memory_count,
            'memory_size_mb': memory_size_mb,
            'disk_images': disk_count, 
            'disk_size_mb': disk_size_mb,
            'cache_dir': cls._cache_dir
        }
            
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
            # double-sided card - check if both images are cached
            front_face = card_faces[0]
            back_face = card_faces[1]
            
            front_image_uris = front_face.get('image_uris', {})
            back_image_uris = back_face.get('image_uris', {})
            
            if 'normal' in front_image_uris and 'normal' in back_image_uris:
                front_url = front_image_uris['normal']
                back_url = back_image_uris['normal']
                
                front_cached = self._get_cached_image(front_url) is not None
                back_cached = self._get_cached_image(back_url) is not None
                
                if front_cached and back_cached:
                    self.display_double_sided_from_cache(card_faces, front_url, back_url)
                    return
                
            # not fully cached yet
            if status == 'loading':
                self.display_loading_message("Loading double-sided card...")
            else:
                self.display_loading_message("Loading images...")
                
        else:
            # single-sided card - check if image is cached
            image_url = None
            
            if 'normal' in image_uris:
                image_url = image_uris['normal']
            elif len(card_faces) >= 1 and 'image_uris' in card_faces[0]:
                face_image_uris = card_faces[0]['image_uris']
                if 'normal' in face_image_uris:
                    image_url = face_image_uris['normal']
                    
            if image_url and self._get_cached_image(image_url) is not None:
                self.display_single_from_cache(image_url)
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
            
    def _get_cached_image(self, image_url: str) -> Image.Image:
        """get image from global cache by url"""
        url_cache_key = self._get_cache_key(image_url)
        with self._cache_lock:
            if url_cache_key in self._global_image_cache:
                self._cache_access_times[url_cache_key] = time.time()
                return self._global_image_cache[url_cache_key]
        return None

    def display_single_from_cache(self, image_url: str):
        """display single card from cache"""
        try:
            image = self._get_cached_image(image_url)
            if image is None:
                self.display_loading_message("Loading image...")
                return
                
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
            
        except Exception as e:
            print(f"error displaying cached image: {e}")
            self.display_no_image()
    
    def display_single_card(self, image_url: str):
        """display a single card image (legacy method - now loads to cache)"""
        try:
            image = self.load_image_to_cache(image_url, f"single_{hash(image_url)}")
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
            
    def display_double_sided_from_cache(self, card_faces: list, front_image_url: str, back_image_url: str):
        """display double-sided card from cache"""
        try:
            front_image = self._get_cached_image(front_image_url)
            back_image = self._get_cached_image(back_image_url)
            
            if front_image is None or back_image is None:
                self.display_loading_message("Loading double-sided card...")
                return
                
            front_face = card_faces[0]
            back_face = card_faces[1]
            
            # load front image from cache
            front_image_copy = front_image.copy()
            front_image_copy.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(front_image_copy)
            
            # load back image from cache  
            back_image_copy = back_image.copy()
            back_image_copy.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image_back = ImageTk.PhotoImage(back_image_copy)
            
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
            
    def display_simple_first_card(self, printing: dict):
        """simple fallback method to display first card without caching"""
        try:
            print("displaying first card in simple mode...")
            card_faces = printing.get('card_faces', [])
            image_uris = printing.get('image_uris', {})
            
            # update card info first
            self.update_card_info_display(printing)
            
            # check if this is a double-sided card
            if len(card_faces) >= 2:
                # double-sided - show both faces
                self.display_double_sided_simple(card_faces)
            elif 'normal' in image_uris:
                self.display_single_card_simple(image_uris['normal'])
            elif len(card_faces) >= 1 and 'image_uris' in card_faces[0]:
                face_image_uris = card_faces[0]['image_uris']
                if 'normal' in face_image_uris:
                    self.display_single_card_simple(face_image_uris['normal'])
                else:
                    self.display_loading_message("No images available")
            else:
                # if we get here, no images found
                self.display_loading_message("No images available")
            
        except Exception as e:
            print(f"error in display_simple_first_card: {e}")
            self.display_loading_message("Ready - use navigation buttons")
            
    def get_image_with_simple_cache(self, image_url: str) -> Image.Image:
        """get image with simple caching if enabled, otherwise download fresh"""
        if self.simple_cache and image_url in self.simple_image_cache:
            print(f"cache hit: {image_url[:50]}...")
            return self.simple_image_cache[image_url]
        
        print(f"downloading: {image_url[:50]}...")
        image_data = self.request_scryfall_image(image_url)
        image = Image.open(BytesIO(image_data))
        
        # cache it if simple caching is enabled
        if self.simple_cache:
            self.simple_image_cache[image_url] = image.copy()
            
        return image
        
    def apply_color_balance(self, image: Image.Image, balance: float) -> Image.Image:
        """apply color temperature adjustment (-100 to +100)"""
        if abs(balance) < 0.01:  # skip if no adjustment needed
            return image
            
        try:
            # convert to rgb if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # create color adjustment - negative = cooler (more blue), positive = warmer (more red/yellow) 
            if balance < 0:  # cooler
                # add blue, reduce red/yellow
                adjustment = abs(balance) / 100.0 * 0.3  # limit max adjustment
                # multiply red and green channels by (1-adjustment), keep blue
                r, g, b = image.split()
                r = r.point(lambda x: int(x * (1 - adjustment)))
                g = g.point(lambda x: int(x * (1 - adjustment * 0.5)))  # less green reduction
                image = Image.merge('RGB', (r, g, b))
            else:  # warmer
                # add red/yellow, reduce blue
                adjustment = balance / 100.0 * 0.3
                r, g, b = image.split()
                b = b.point(lambda x: int(x * (1 - adjustment)))
                g = g.point(lambda x: int(x * (1 - adjustment * 0.3)))  # slight green reduction
                image = Image.merge('RGB', (r, g, b))
                
            return image
        except Exception as e:
            print(f"color balance error: {e}")
            return image
        
    def apply_gamma_correction(self, image: Image.Image, gamma: float) -> Image.Image:
        """apply gamma correction"""
        if abs(gamma - 1.0) < 0.01:  # skip if no adjustment needed
            return image
            
        # ensure image is in rgb mode for consistent processing
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # build lookup table for gamma correction (for 8-bit values)
        gamma_table = [int(((i / 255.0) ** (1.0 / gamma)) * 255) for i in range(256)]
        
        # apply gamma correction
        try:
            return image.point(gamma_table * 3)  # multiply by 3 for r, g, b channels
        except Exception as e:
            print(f"gamma correction fallback: {e}")
            # fallback: convert to numpy and back if available, or return original
            return image
        
    def create_selection_mask(self, image: Image.Image) -> Image.Image:
        """create a new empty selection mask for the given image"""
        return Image.new('L', image.size, 0)  # black = not selected
        
    def flood_fill_selection(self, image: Image.Image, x: int, y: int, tolerance: int) -> Image.Image:
        """create selection mask using flood fill from clicked point"""
        try:
            # convert coordinates from display to actual image coordinates
            img_width, img_height = image.size
            
            # ensure click is within bounds
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                return Image.new('L', image.size, 0)
            
            # get target color
            if image.mode != 'RGB':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image
                
            target_color = rgb_image.getpixel((x, y))
            
            # create mask using numpy for efficiency
            img_array = np.array(rgb_image)
            mask_array = np.zeros((img_height, img_width), dtype=np.uint8)
            
            # calculate color difference
            diff = np.sqrt(np.sum((img_array - target_color) ** 2, axis=2))
            mask_array[diff <= tolerance] = 255  # white = selected
            
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            print(f"flood fill error: {e}")
            return Image.new('L', image.size, 0)
    
    def apply_selection_to_mask(self, x: int, y: int, is_front: bool = True):
        """apply brush selection at given coordinates"""
        try:
            if is_front and self.selection_mask:
                mask = self.selection_mask
            elif not is_front and self.selection_mask_back:
                mask = self.selection_mask_back
            else:
                return
                
            # create drawing context
            draw = ImageDraw.Draw(mask)
            brush_radius = self.brush_size // 2
            
            # determine fill color based on current drawing mode
            mode_to_use = self.current_draw_mode if self.is_drawing else self.brush_mode
            fill_color = 255 if mode_to_use == 'add' else 0  # white = selected, black = not selected
            
            # draw circle at position
            draw.ellipse([
                x - brush_radius, y - brush_radius,
                x + brush_radius, y + brush_radius
            ], fill=fill_color)
            
            # if we have a last position, draw line between them
            if self.last_draw_pos:
                last_x, last_y = self.last_draw_pos
                draw.line([last_x, last_y, x, y], fill=fill_color, width=self.brush_size)
                
            self.last_draw_pos = (x, y)
            
        except Exception as e:
            print(f"brush selection error: {e}")
            
    def clear_selection(self):
        """clear all selections"""
        if self.original_image:
            self.selection_mask = self.create_selection_mask(self.original_image)
        if self.original_image_back:
            self.selection_mask_back = self.create_selection_mask(self.original_image_back)
        self.update_selection_display()
        self.refresh_image_display()

    def apply_image_adjustments(self, image: Image.Image, mask: Image.Image = None) -> Image.Image:
        """apply all current adjustments to an image, optionally masked"""
        if not image:
            return None
            
        try:
            # start with copy of original and ensure RGB mode
            original = image.copy()
            if original.mode != 'RGB':
                original = original.convert('RGB')
            
            # create adjusted version
            adjusted = original.copy()
            
            # apply gamma correction first (affects overall brightness curve)
            adjusted = self.apply_gamma_correction(adjusted, self.gamma)
            
            # apply brightness
            if abs(self.brightness - 1.0) > 0.01:
                enhancer = ImageEnhance.Brightness(adjusted)
                adjusted = enhancer.enhance(self.brightness)
                
            # apply contrast  
            if abs(self.contrast - 1.0) > 0.01:
                enhancer = ImageEnhance.Contrast(adjusted)
                adjusted = enhancer.enhance(self.contrast)
                
            # apply saturation
            if abs(self.saturation - 1.0) > 0.01:
                enhancer = ImageEnhance.Color(adjusted)
                adjusted = enhancer.enhance(self.saturation)
                
            # apply color balance
            adjusted = self.apply_color_balance(adjusted, self.color_balance)
            
            # if we have a mask, composite adjusted and original based on mask
            if mask:
                # mask should be same size as image
                if mask.size != original.size:
                    mask = mask.resize(original.size, Image.Resampling.LANCZOS)
                    
                # composite: where mask is white (255), use adjusted; where black (0), use original
                result = Image.composite(adjusted, original, mask)
                return result
            else:
                return adjusted
            
        except Exception as e:
            print(f"error applying image adjustments: {e}")
            return image  # return original if adjustment fails
            
    def update_selection_display(self):
        """update the visual display of selections (overlay on images)"""
        # this will be implemented after we add the canvas overlays
        pass

    def display_single_card_simple(self, image_url: str):
        """display a single card image with optional simple caching and adjustments"""
        try:
            image = self.get_image_with_simple_cache(image_url)
            
            # store original for adjustments
            self.original_image = image.copy()
            self.original_image_back = None
            
            # initialize selection mask if needed
            if self.selection_mask is None:
                self.selection_mask = self.create_selection_mask(image)
            
            # apply adjustments with mask
            mask_to_use = self.selection_mask if self.selection_mode != 'none' else None
            adjusted_image = self.apply_image_adjustments(image, mask_to_use)
            adjusted_image.thumbnail((300, 420), Image.Resampling.LANCZOS)
            
            self.photo_image = ImageTk.PhotoImage(adjusted_image)
            
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
            print(f"error loading single image: {e}")
            self.image_label.configure(text="image failed to load", image="")
            
    def display_double_sided_simple(self, card_faces: list):
        """display both sides of a double-sided card with optional simple caching and adjustments"""
        try:
            front_face = card_faces[0]
            back_face = card_faces[1]
            
            front_image_uris = front_face.get('image_uris', {})
            back_image_uris = back_face.get('image_uris', {})
            
            if 'normal' not in front_image_uris or 'normal' not in back_image_uris:
                self.image_label.configure(text="no images available", image="")
                return
                
            print(f"loading double-sided images...")
            
            # load front image with simple caching
            front_image = self.get_image_with_simple_cache(front_image_uris['normal'])
            back_image = self.get_image_with_simple_cache(back_image_uris['normal'])
            
            # store originals for adjustments
            self.original_image = front_image.copy()
            self.original_image_back = back_image.copy()
            
            # initialize selection masks if needed
            if self.selection_mask is None:
                self.selection_mask = self.create_selection_mask(front_image)
            if self.selection_mask_back is None:
                self.selection_mask_back = self.create_selection_mask(back_image)
            
            # apply adjustments and resize with masks
            front_mask = self.selection_mask if self.selection_mode != 'none' else None
            back_mask = self.selection_mask_back if self.selection_mode != 'none' else None
            
            adjusted_front = self.apply_image_adjustments(front_image, front_mask)
            adjusted_front.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(adjusted_front)
            
            adjusted_back = self.apply_image_adjustments(back_image, back_mask)
            adjusted_back.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image_back = ImageTk.PhotoImage(adjusted_back)
            
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
            self.image_label.configure(text="double-sided images failed to load", image="")
            
    def refresh_image_display(self):
        """refresh the image display with current adjustments and selections"""
        try:
            if self.original_image and self.original_image_back:
                # double-sided card
                front_mask = self.selection_mask if self.selection_mode != 'none' else None
                back_mask = self.selection_mask_back if self.selection_mode != 'none' else None
                
                adjusted_front = self.apply_image_adjustments(self.original_image, front_mask)
                adjusted_front.thumbnail((250, 350), Image.Resampling.LANCZOS)
                self.photo_image = ImageTk.PhotoImage(adjusted_front)
                
                adjusted_back = self.apply_image_adjustments(self.original_image_back, back_mask)  
                adjusted_back.thumbnail((250, 350), Image.Resampling.LANCZOS)
                self.photo_image_back = ImageTk.PhotoImage(adjusted_back)
                
                # update display
                self.image_label.configure(image=self.photo_image)
                if hasattr(self, 'image_label_back'):
                    self.image_label_back.configure(image=self.photo_image_back)
                    
            elif self.original_image:
                # single card
                mask = self.selection_mask if self.selection_mode != 'none' else None
                adjusted_image = self.apply_image_adjustments(self.original_image, mask)
                adjusted_image.thumbnail((300, 420), Image.Resampling.LANCZOS)
                self.photo_image = ImageTk.PhotoImage(adjusted_image)
                
                # update display
                self.image_label.configure(image=self.photo_image)
                
        except Exception as e:
            print(f"error refreshing image display: {e}")
            
    def reset_adjustments(self):
        """reset all adjustments to default values"""
        self.brightness = 1.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.gamma = 1.0
        self.color_balance = 0.0
        
        # update sliders if they exist
        if hasattr(self, 'brightness_var'):
            self.brightness_var.set(self.brightness)
        if hasattr(self, 'contrast_var'):
            self.contrast_var.set(self.contrast)
        if hasattr(self, 'saturation_var'):
            self.saturation_var.set(self.saturation)
        if hasattr(self, 'gamma_var'):
            self.gamma_var.set(self.gamma)
        if hasattr(self, 'color_balance_var'):
            self.color_balance_var.set(self.color_balance)
            
        # refresh display
        self.refresh_image_display()
            
    def update_card_info_display(self, printing: dict):
        """update just the card info section"""
        try:
            # update counter label with current index
            counter_text = f"{self.current_index + 1} / {len(self.printings)}"
            self.counter_label.configure(text=counter_text)
            
            # update card info (reuse existing logic)
            set_name = printing.get('set_name', 'Unknown Set')
            set_code = printing.get('set', '').upper()
            collector_number = printing.get('collector_number', '')
            rarity = printing.get('rarity', '').title()
            release_date = printing.get('released_at', '')
            
            info_text = f"Set: {set_name} ({set_code})\n"
            info_text += f"Collector #: {collector_number}\n"
            info_text += f"Rarity: {rarity}\n"
            info_text += f"Released: {release_date}"
            
            self.info_label.configure(text=info_text)
            
        except Exception as e:
            print(f"error updating card info: {e}")
        
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
        
        # bind mouse events for selection on back image
        self.image_label_back.bind("<Button-1>", self.on_image_click)
        self.image_label_back.bind("<B1-Motion>", self.on_image_drag)
        self.image_label_back.bind("<ButtonRelease-1>", self.on_image_release)
            
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
            
            if self.use_caching:
                self.update_display()
                # trigger background loading for this printing if not loaded
                with self.loading_lock:
                    if self.current_index not in self.loading_status:
                        self.executor.submit(self.background_load_printing, self.current_index)
            else:
                # simple mode - just display current card directly
                self.display_simple_current_card()
            
    def next_card(self):
        """go to next printing"""
        if self.printings and self.current_index < len(self.printings) - 1:
            self.current_index += 1
            
            if self.use_caching:
                self.update_display()
                # trigger background loading for this printing if not loaded
                with self.loading_lock:
                    if self.current_index not in self.loading_status:
                        self.executor.submit(self.background_load_printing, self.current_index)
            else:
                # simple mode - just display current card directly
                self.display_simple_current_card()
                
    def display_simple_current_card(self):
        """display the current card in simple mode without caching"""
        if not self.printings or self.current_index >= len(self.printings):
            return
            
        printing = self.printings[self.current_index]
        
        # update card info
        self.update_card_info_display(printing)
        
        # display the image(s)
        card_faces = printing.get('card_faces', [])
        image_uris = printing.get('image_uris', {})
        
        # check if this is a double-sided card
        if len(card_faces) >= 2:
            # double-sided card - show both faces
            self.display_double_sided_simple(card_faces)
        elif 'normal' in image_uris:
            self.display_single_card_simple(image_uris['normal'])
        elif len(card_faces) >= 1 and 'image_uris' in card_faces[0]:
            face_image_uris = card_faces[0]['image_uris']
            if 'normal' in face_image_uris:
                self.display_single_card_simple(face_image_uris['normal'])
            else:
                self.image_label.configure(text="no image available", image="")
        else:
            # no image found
            self.image_label.configure(text="no image available", image="")
            
    def select_card(self):
        """select current printing and close window"""
        if self.printings:
            self.selected_printing = self.printings[self.current_index]
            
            # also store the adjusted images for saving (with selection masks if active)
            if self.original_image and self.original_image_back:
                # double-sided card - store both adjusted images with masks
                front_mask = self.selection_mask if self.selection_mode != 'none' else None
                back_mask = self.selection_mask_back if self.selection_mode != 'none' else None
                
                self.selected_printing['adjusted_front_image'] = self.apply_image_adjustments(self.original_image, front_mask)
                self.selected_printing['adjusted_back_image'] = self.apply_image_adjustments(self.original_image_back, back_mask)
            elif self.original_image:
                # single card - store adjusted image with mask
                mask = self.selection_mask if self.selection_mode != 'none' else None
                self.selected_printing['adjusted_image'] = self.apply_image_adjustments(self.original_image, mask)
                
        self.close_gui()
        
    def skip_card(self):
        """use first available printing (skip manual selection)"""
        if self.printings:
            self.selected_printing = self.printings[0]
            # no adjustments when skipping - use original scryfall images
            print(f"using first available printing for card (no adjustments)")
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
            
        # clear per-dialog state (but keep global image cache for next time!)
        self.loading_status.clear()
        
    def create_adjustment_controls(self, parent):
        """create the adjustment control sliders and buttons"""
        row = 0
        
        # === SELECTION TOOLS SECTION ===
        selection_frame = ttk.LabelFrame(parent, text="Selection Tools", padding="10")
        selection_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        selection_frame.columnconfigure(0, weight=1)
        row += 1
        
        # selection mode radio buttons
        self.selection_mode_var = tk.StringVar(value='none')
        
        none_radio = ttk.Radiobutton(selection_frame, text="No Selection (Adjust All)", 
                                   variable=self.selection_mode_var, value='none',
                                   command=self.on_selection_mode_change)
        none_radio.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        brush_radio = ttk.Radiobutton(selection_frame, text="Brush Selection", 
                                    variable=self.selection_mode_var, value='brush',
                                    command=self.on_selection_mode_change)
        brush_radio.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        color_radio = ttk.Radiobutton(selection_frame, text="Magic Wand (Color Select)", 
                                    variable=self.selection_mode_var, value='color',
                                    command=self.on_selection_mode_change)
        color_radio.grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        
        # brush controls (only visible when brush mode is active)
        self.brush_controls_frame = ttk.Frame(selection_frame)
        self.brush_controls_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # brush mode radio buttons
        self.brush_mode_var = tk.StringVar(value='add')
        add_brush_radio = ttk.Radiobutton(self.brush_controls_frame, text="Add to Selection", 
                                        variable=self.brush_mode_var, value='add',
                                        command=self.on_brush_mode_change)
        add_brush_radio.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        subtract_brush_radio = ttk.Radiobutton(self.brush_controls_frame, text="Erase Selection", 
                                             variable=self.brush_mode_var, value='subtract',
                                             command=self.on_brush_mode_change)
        subtract_brush_radio.grid(row=0, column=1, sticky=tk.W, padx=(20, 0), pady=(0, 5))
        
        # brush size
        self.brush_size_frame = ttk.Frame(self.brush_controls_frame)
        self.brush_size_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 0))
        
        ttk.Label(self.brush_size_frame, text="Brush Size:").grid(row=0, column=0, sticky=tk.W)
        self.brush_size_var = tk.IntVar(value=self.brush_size)
        brush_scale = ttk.Scale(self.brush_size_frame, from_=5, to=50, variable=self.brush_size_var, 
                              orient=tk.HORIZONTAL, length=120, command=self.on_brush_size_change)
        brush_scale.grid(row=0, column=1, padx=(10, 0))
        self.brush_size_label = ttk.Label(self.brush_size_frame, text=str(self.brush_size))
        self.brush_size_label.grid(row=0, column=2, padx=(10, 0))
        
        # color tolerance (only visible when color mode is active)
        self.tolerance_frame = ttk.Frame(selection_frame)
        self.tolerance_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(self.tolerance_frame, text="Color Tolerance:").grid(row=0, column=0, sticky=tk.W)
        self.tolerance_var = tk.IntVar(value=self.color_tolerance)
        tolerance_scale = ttk.Scale(self.tolerance_frame, from_=5, to=100, variable=self.tolerance_var, 
                                  orient=tk.HORIZONTAL, length=120, command=self.on_tolerance_change)
        tolerance_scale.grid(row=0, column=1, padx=(10, 0))
        self.tolerance_label = ttk.Label(self.tolerance_frame, text=str(self.color_tolerance))
        self.tolerance_label.grid(row=0, column=2, padx=(10, 0))
        
        # selection control buttons
        button_frame = ttk.Frame(selection_frame)
        button_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        clear_sel_button = ttk.Button(button_frame, text="Clear Selection", command=self.clear_selection)
        clear_sel_button.grid(row=0, column=0, padx=(0, 5))
        
        select_all_button = ttk.Button(button_frame, text="Select All", command=self.select_all)
        select_all_button.grid(row=0, column=1, padx=(0, 5))
        
        frame_sel_button = ttk.Button(button_frame, text="Frame Only", command=self.select_frame)
        frame_sel_button.grid(row=0, column=2, padx=(0, 5))
        
        invert_sel_button = ttk.Button(button_frame, text="Invert", command=self.invert_selection)
        invert_sel_button.grid(row=0, column=3)
        
        # smart fix button (prominent)
        smart_fix_button = ttk.Button(button_frame, text="🎯 Smart Fix", command=self.smart_fix_card)
        smart_fix_button.grid(row=1, column=0, columnspan=4, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # instructions label
        self.instruction_label = ttk.Label(selection_frame, text="Try Smart Fix first, then fine-tune", 
                                         font=("Arial", 9), foreground="gray")
        self.instruction_label.grid(row=7, column=0, sticky=tk.W, pady=(10, 0))
        
        # update initial visibility
        self.update_selection_ui_visibility()
        
        # === IMAGE ADJUSTMENTS SECTION ===
        adj_frame = ttk.LabelFrame(parent, text="Image Adjustments", padding="10")
        adj_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        adj_frame.columnconfigure(0, weight=1)
        row += 1
        
        adj_row = 0
        
        # brightness control
        ttk.Label(adj_frame, text="Brightness:", font=("Arial", 10, "bold")).grid(row=adj_row, column=0, sticky=tk.W, pady=(0, 5))
        adj_row += 1
        self.brightness_var = tk.DoubleVar(value=self.brightness)
        brightness_scale = ttk.Scale(adj_frame, from_=0.3, to=2.0, variable=self.brightness_var, 
                                   orient=tk.HORIZONTAL, length=180,
                                   command=self.on_brightness_change)
        brightness_scale.grid(row=adj_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.brightness_value_label = ttk.Label(adj_frame, text=f"{self.brightness:.2f}")
        self.brightness_value_label.grid(row=adj_row, column=1, padx=(10, 0), pady=(0, 10))
        adj_row += 1
        
        # contrast control  
        ttk.Label(adj_frame, text="Contrast:", font=("Arial", 10, "bold")).grid(row=adj_row, column=0, sticky=tk.W, pady=(0, 5))
        adj_row += 1
        self.contrast_var = tk.DoubleVar(value=self.contrast)
        contrast_scale = ttk.Scale(adj_frame, from_=0.3, to=2.5, variable=self.contrast_var, 
                                 orient=tk.HORIZONTAL, length=180,
                                 command=self.on_contrast_change)
        contrast_scale.grid(row=adj_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.contrast_value_label = ttk.Label(adj_frame, text=f"{self.contrast:.2f}")
        self.contrast_value_label.grid(row=adj_row, column=1, padx=(10, 0), pady=(0, 10))
        adj_row += 1
        
        # saturation control
        ttk.Label(adj_frame, text="Saturation:", font=("Arial", 10, "bold")).grid(row=adj_row, column=0, sticky=tk.W, pady=(0, 5))
        adj_row += 1
        self.saturation_var = tk.DoubleVar(value=self.saturation)
        saturation_scale = ttk.Scale(adj_frame, from_=0.0, to=2.0, variable=self.saturation_var, 
                                   orient=tk.HORIZONTAL, length=180,
                                   command=self.on_saturation_change)
        saturation_scale.grid(row=adj_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.saturation_value_label = ttk.Label(adj_frame, text=f"{self.saturation:.2f}")
        self.saturation_value_label.grid(row=adj_row, column=1, padx=(10, 0), pady=(0, 10))
        adj_row += 1
        
        # gamma control
        ttk.Label(adj_frame, text="Gamma:", font=("Arial", 10, "bold")).grid(row=adj_row, column=0, sticky=tk.W, pady=(0, 5))
        adj_row += 1
        self.gamma_var = tk.DoubleVar(value=self.gamma)
        gamma_scale = ttk.Scale(adj_frame, from_=0.3, to=2.5, variable=self.gamma_var, 
                              orient=tk.HORIZONTAL, length=180,
                              command=self.on_gamma_change)
        gamma_scale.grid(row=adj_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.gamma_value_label = ttk.Label(adj_frame, text=f"{self.gamma:.2f}")
        self.gamma_value_label.grid(row=adj_row, column=1, padx=(10, 0), pady=(0, 10))
        adj_row += 1
        
        # color balance control
        ttk.Label(adj_frame, text="Color Temperature:", font=("Arial", 10, "bold")).grid(row=adj_row, column=0, sticky=tk.W, pady=(0, 5))
        adj_row += 1
        ttk.Label(adj_frame, text="(Cool ← → Warm)", font=("Arial", 8)).grid(row=adj_row, column=0, sticky=tk.W, pady=(0, 5))
        adj_row += 1
        self.color_balance_var = tk.DoubleVar(value=self.color_balance)
        color_balance_scale = ttk.Scale(adj_frame, from_=-100, to=100, variable=self.color_balance_var, 
                                      orient=tk.HORIZONTAL, length=180,
                                      command=self.on_color_balance_change)
        color_balance_scale.grid(row=adj_row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.color_balance_value_label = ttk.Label(adj_frame, text=f"{self.color_balance:.0f}")
        self.color_balance_value_label.grid(row=adj_row, column=1, padx=(10, 0), pady=(0, 10))
        adj_row += 1
        
        # separator
        ttk.Separator(adj_frame, orient=tk.HORIZONTAL).grid(row=adj_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        adj_row += 1
        
        # reset button
        reset_button = ttk.Button(adj_frame, text="Reset All", command=self.reset_adjustments)
        reset_button.grid(row=adj_row, column=0, columnspan=2, pady=(0, 10))
        adj_row += 1
        
        # tips label
        tips_text = "Tips for MTG cards:\n• Increase contrast for borders\n• Adjust gamma for text areas\n• Warm colors for older sets\n• Cool colors for modern sets"
        tips_label = ttk.Label(adj_frame, text=tips_text, font=("Arial", 9), justify=tk.LEFT, 
                              foreground="gray", wraplength=200)
        tips_label.grid(row=adj_row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # configure column weights  
        parent.columnconfigure(0, weight=1)
        
    def select_all(self):
        """select entire image"""
        if self.original_image:
            self.selection_mask = Image.new('L', self.original_image.size, 255)  # white = selected
        if self.original_image_back:
            self.selection_mask_back = Image.new('L', self.original_image_back.size, 255)
        self.refresh_image_display()
        
    def select_frame(self):
        """select only the card frame areas (excludes art and text)"""
        if self.original_image:
            self.selection_mask = self.create_smart_frame_mask(self.original_image)
        if self.original_image_back:
            self.selection_mask_back = self.create_smart_frame_mask(self.original_image_back)
        self.refresh_image_display()
        print("Frame areas selected - adjust colors without affecting art or text")
        
    def invert_selection(self):
        """invert current selection"""
        try:
            if self.original_image and self.selection_mask:
                # invert selection mask (255 - pixel_value)
                inverted_array = 255 - np.array(self.selection_mask)
                self.selection_mask = Image.fromarray(inverted_array, mode='L')
                
            if self.original_image_back and self.selection_mask_back:
                inverted_array = 255 - np.array(self.selection_mask_back)
                self.selection_mask_back = Image.fromarray(inverted_array, mode='L')
                
            self.refresh_image_display()
            
        except Exception as e:
            print(f"selection inversion error: {e}")
            
    def analyze_card_issues(self, image: Image.Image) -> dict:
        """analyze image for common MTG card issues"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            issues = {
                'text_box_washed_out': False,
                'color_cast': 'neutral',
                'overall_too_dark': False,
                'overall_too_bright': False,
                'low_contrast': False,
                'text_box_region': None,
                'border_region': None
            }
            
            # detect text box region (usually bottom 35-45% of card)
            text_box_start = int(height * 0.55)
            text_box_region = img_array[text_box_start:, :]
            border_top_region = img_array[:int(height * 0.15), :]
            border_side_region = np.concatenate([
                img_array[:, :int(width * 0.1)],
                img_array[:, int(width * 0.9):]
            ], axis=1)
            
            issues['text_box_region'] = (text_box_start, height, 0, width)
            issues['border_region'] = border_top_region
            
            # analyze brightness levels
            text_box_brightness = np.mean(text_box_region)
            border_brightness = np.mean(border_top_region)
            overall_brightness = np.mean(img_array)
            
            # detect washed out text box (significantly brighter than border)
            if text_box_brightness > border_brightness + 20:
                issues['text_box_washed_out'] = True
                
            # analyze color temperature
            r_avg = np.mean(img_array[:, :, 0])
            g_avg = np.mean(img_array[:, :, 1])
            b_avg = np.mean(img_array[:, :, 2])
            
            warm_score = (r_avg + g_avg) - b_avg * 2
            if warm_score > 20:
                issues['color_cast'] = 'warm'
            elif warm_score < -20:
                issues['color_cast'] = 'cool'
                
            # overall brightness issues
            if overall_brightness < 80:
                issues['overall_too_dark'] = True
            elif overall_brightness > 180:
                issues['overall_too_bright'] = True
                
            # contrast analysis
            contrast_score = np.std(img_array)
            if contrast_score < 45:
                issues['low_contrast'] = True
                
            return issues
            
        except Exception as e:
            print(f"card analysis error: {e}")
            return {'error': True}
            
    def create_text_box_mask(self, image: Image.Image) -> Image.Image:
        """create mask for the text box area"""
        try:
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            
            # create rectangular mask for text box region
            text_box_start = int(height * 0.55)
            text_box_end = int(height * 0.92)  # leave some margin at bottom
            
            mask_array = np.array(mask)
            mask_array[text_box_start:text_box_end, :] = 255
            
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            print(f"text box mask error: {e}")
            return Image.new('L', image.size, 0)
            
    def create_border_mask(self, image: Image.Image) -> Image.Image:
        """create mask for the entire card border/frame (excludes art area)"""
        try:
            width, height = image.size
            mask = Image.new('L', (width, height), 255)  # start with everything selected
            
            mask_array = np.array(mask)
            
            # define art area (center rectangle) - exclude from border mask
            art_left = int(width * 0.08)    # 8% from left
            art_right = int(width * 0.92)   # 8% from right  
            art_top = int(height * 0.15)    # 15% from top (below name)
            art_bottom = int(height * 0.55) # 55% from top (above text)
            
            # clear art area (set to black = not selected)
            mask_array[art_top:art_bottom, art_left:art_right] = 0
            
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            print(f"border mask error: {e}")
            return Image.new('L', image.size, 255)
            
    def create_smart_frame_mask(self, image: Image.Image) -> Image.Image:
        """create intelligent mask for MTG card frame areas needing color correction"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            width, height = image.size
            img_array = np.array(image)
            mask = Image.new('L', (width, height), 0)  # start with nothing selected
            mask_array = np.array(mask)
            
            print("Creating precise frame mask...")
            
            # define precise regions that need color correction
            # 1. outer border areas (thin strips around the edges)
            border_width = int(width * 0.08)   # 8% border width
            border_height = int(height * 0.05) # 5% border height
            
            # top border (but not too deep - avoid name area)
            mask_array[0:border_height, :] = 255
            
            # bottom border 
            mask_array[height-border_height:height, :] = 255
            
            # left border
            mask_array[:, 0:border_width] = 255
            
            # right border  
            mask_array[:, width-border_width:width] = 255
            
            # 2. text box background area (bottom portion of card)
            text_box_start = int(height * 0.58)  # start lower to avoid type line
            text_box_end = int(height * 0.90)    # end before bottom border
            text_box_left = int(width * 0.08)    # respect side borders
            text_box_right = int(width * 0.92)   # respect side borders
            
            # select text box background but exclude very bright areas (actual text)
            text_region = img_array[text_box_start:text_box_end, text_box_left:text_box_right]
            brightness = 0.299 * text_region[:, :, 0] + 0.587 * text_region[:, :, 1] + 0.114 * text_region[:, :, 2]
            
            # create text box mask - select background but not bright text
            text_box_mask = np.ones(brightness.shape, dtype=np.uint8) * 255
            text_box_mask[brightness > 200] = 0  # exclude bright text pixels
            
            # apply text box mask
            mask_array[text_box_start:text_box_end, text_box_left:text_box_right] = text_box_mask
            
            # 3. EXCLUDE name box and type line areas completely
            name_box_start = int(height * 0.04)
            name_box_end = int(height * 0.14)
            mask_array[name_box_start:name_box_end, :] = 0  # never select name area
            
            type_line_start = int(height * 0.47) 
            type_line_end = int(height * 0.57)
            mask_array[type_line_start:type_line_end, :] = 0  # never select type line area
            
            # 4. EXCLUDE art area (center)
            art_left = int(width * 0.08)
            art_right = int(width * 0.92) 
            art_top = int(height * 0.14)     # start after name area
            art_bottom = int(height * 0.47)  # end before type line
            
            mask_array[art_top:art_bottom, art_left:art_right] = 0  # never select art
            
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            print(f"smart frame mask error: {e}")
            # fallback to simple text box only
            return self.create_text_box_mask(image)
            
    def create_washed_out_mask(self, image: Image.Image, threshold: int = 200) -> Image.Image:
        """create mask for overly bright/washed out areas"""
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_array = np.array(image)
            
            # calculate brightness (luminance)
            brightness = 0.299 * img_array[:, :, 0] + 0.587 * img_array[:, :, 1] + 0.114 * img_array[:, :, 2]
            
            # create mask where brightness exceeds threshold
            mask_array = np.zeros(brightness.shape, dtype=np.uint8)
            mask_array[brightness > threshold] = 255
            
            return Image.fromarray(mask_array, mode='L')
            
        except Exception as e:
            print(f"washed out mask error: {e}")
            return Image.new('L', image.size, 0)

    def smart_fix_card(self):
        """automatically detect and fix common MTG card issues"""
        try:
            print("🎯 Running Smart Fix analysis...")
            
            # work on front image primarily
            if not self.original_image:
                print("No image loaded for smart fix")
                return
                
            # analyze issues
            issues = self.analyze_card_issues(self.original_image)
            if 'error' in issues:
                print("Smart fix analysis failed")
                return
                
            print(f"Issues detected: {issues}")
            
            # reset to default adjustments first
            self.brightness = 1.0
            self.contrast = 1.0
            self.saturation = 1.0
            self.gamma = 1.0
            self.color_balance = 0.0
            
            # create smart selection mask
            smart_mask = None
            adjustments_applied = []
            
            # fix washed out frame areas (including consistent border treatment)
            if issues['text_box_washed_out']:
                print("Fixing washed out frame areas...")
                frame_mask = self.create_smart_frame_mask(self.original_image)
                smart_mask = self.combine_masks(smart_mask, frame_mask)
                
                # apply corrections for frame areas
                self.brightness = 0.8  # darken
                self.contrast = 1.3    # increase contrast
                self.gamma = 0.9       # adjust midtones
                adjustments_applied.append("frame areas darkened")
                
            # fix color temperature
            if issues['color_cast'] == 'warm':
                print("Correcting warm color cast...")
                self.color_balance = -25  # cool it down
                adjustments_applied.append("cooled color temperature")
                
            elif issues['color_cast'] == 'cool':
                print("Correcting cool color cast...")
                self.color_balance = 15   # warm it up
                adjustments_applied.append("warmed color temperature")
                
            # fix overall brightness issues
            if issues['overall_too_dark']:
                print("Brightening overall image...")
                self.brightness = max(self.brightness, 1.2)
                adjustments_applied.append("brightened image")
                
            elif issues['overall_too_bright']:
                print("Darkening overall image...")
                self.brightness = min(self.brightness, 0.85)
                adjustments_applied.append("darkened image")
                
            # fix low contrast
            if issues['low_contrast']:
                print("Boosting contrast...")
                self.contrast = max(self.contrast, 1.4)
                adjustments_applied.append("boosted contrast")
                
            # enhance saturation if colors are washed out
            if issues['text_box_washed_out'] or issues['overall_too_bright']:
                print("Enhancing color saturation...")
                self.saturation = 1.15
                adjustments_applied.append("enhanced saturation")
                
            # update slider values
            if hasattr(self, 'brightness_var'):
                self.brightness_var.set(self.brightness)
                self.brightness_value_label.configure(text=f"{self.brightness:.2f}")
            if hasattr(self, 'contrast_var'):
                self.contrast_var.set(self.contrast)  
                self.contrast_value_label.configure(text=f"{self.contrast:.2f}")
            if hasattr(self, 'saturation_var'):
                self.saturation_var.set(self.saturation)
                self.saturation_value_label.configure(text=f"{self.saturation:.2f}")
            if hasattr(self, 'gamma_var'):
                self.gamma_var.set(self.gamma)
                self.gamma_value_label.configure(text=f"{self.gamma:.2f}")
            if hasattr(self, 'color_balance_var'):
                self.color_balance_var.set(self.color_balance)
                self.color_balance_value_label.configure(text=f"{self.color_balance:.0f}")
                
            # apply smart mask if we created one
            if smart_mask:
                self.selection_mask = smart_mask
                
                # if double-sided card, apply same mask to back
                if self.original_image_back:
                    self.selection_mask_back = self.create_smart_frame_mask(self.original_image_back)
                    print("Smart frame mask applied to back side as well")
                
                self.selection_mode = 'color'  # switch to show we have a selection
                if hasattr(self, 'selection_mode_var'):
                    self.selection_mode_var.set('color')
                print("Smart selection mask applied")
                
            # update display
            self.refresh_image_display()
            
            # show results
            if adjustments_applied:
                print(f"✅ Smart Fix complete! Applied: {', '.join(adjustments_applied)}")
                print("💡 Fine-tune with sliders or add more selections if needed")
            else:
                print("✅ Image looks good - no major issues detected")
                
        except Exception as e:
            print(f"Smart fix error: {e}")
        
    def update_selection_ui_visibility(self):
        """show/hide selection controls based on current mode"""
        if hasattr(self, 'brush_controls_frame'):
            if self.selection_mode == 'brush':
                self.brush_controls_frame.grid()
                self.tolerance_frame.grid_remove()
                self.instruction_label.configure(text="Paint to add/erase selection • Perfect for fine-tuning Smart Fix")
            elif self.selection_mode == 'color':
                self.brush_controls_frame.grid_remove()
                self.tolerance_frame.grid()
                self.instruction_label.configure(text="Click color to select • Cmd/Ctrl+Click to add • Alt/Shift+Click to subtract")
            else:
                self.brush_controls_frame.grid_remove()
                self.tolerance_frame.grid_remove()
                self.instruction_label.configure(text="Try Smart Fix first, then fine-tune")
        
    def on_selection_mode_change(self):
        """called when selection mode radio button changes"""
        self.selection_mode = self.selection_mode_var.get()
        self.update_selection_ui_visibility()
        print(f"selection mode: {self.selection_mode}")
        
    def on_brush_size_change(self, value):
        """called when brush size slider changes"""
        self.brush_size = int(float(value))
        self.brush_size_label.configure(text=str(self.brush_size))
        
    def on_brush_mode_change(self):
        """called when brush mode radio button changes"""
        self.brush_mode = self.brush_mode_var.get()
        print(f"brush mode: {self.brush_mode}")
        
    def on_tolerance_change(self, value):
        """called when color tolerance slider changes"""
        self.color_tolerance = int(float(value))
        self.tolerance_label.configure(text=str(self.color_tolerance))

    def on_brightness_change(self, value):
        """called when brightness slider changes"""
        self.brightness = float(value)
        self.brightness_value_label.configure(text=f"{self.brightness:.2f}")
        self.refresh_image_display()
        
    def on_contrast_change(self, value):
        """called when contrast slider changes"""
        self.contrast = float(value)
        self.contrast_value_label.configure(text=f"{self.contrast:.2f}")
        self.refresh_image_display()
        
    def on_saturation_change(self, value):
        """called when saturation slider changes"""
        self.saturation = float(value)
        self.saturation_value_label.configure(text=f"{self.saturation:.2f}")
        self.refresh_image_display()
        
    def on_gamma_change(self, value):
        """called when gamma slider changes"""
        self.gamma = float(value)
        self.gamma_value_label.configure(text=f"{self.gamma:.2f}")
        self.refresh_image_display()
        
    def on_color_balance_change(self, value):
        """called when color balance slider changes"""
        self.color_balance = float(value)
        self.color_balance_value_label.configure(text=f"{self.color_balance:.0f}")
        self.refresh_image_display()
        
    def get_image_coordinates(self, event, is_front=True):
        """convert screen coordinates to image coordinates"""
        try:
            # get current image and its display size
            if is_front and self.original_image:
                original_size = self.original_image.size
                # for single cards, display size is (300, 420) max
                if self.original_image_back:
                    display_max = (250, 350)  # double-sided
                else:
                    display_max = (300, 420)  # single
            elif not is_front and self.original_image_back:
                original_size = self.original_image_back.size
                display_max = (250, 350)  # back side of double-sided card
            else:
                return None, None
                
            # calculate actual display size (thumbnail preserves aspect ratio)
            img_width, img_height = original_size
            max_width, max_height = display_max
            
            # calculate scale factor (same as thumbnail)
            scale = min(max_width / img_width, max_height / img_height)
            display_width = int(img_width * scale)
            display_height = int(img_height * scale)
            
            # convert event coordinates to image coordinates
            # event coordinates are relative to the label widget
            label_width = event.widget.winfo_width()
            label_height = event.widget.winfo_height()
            
            # center the image within the label
            x_offset = (label_width - display_width) // 2
            y_offset = (label_height - display_height) // 2
            
            # adjust coordinates
            img_x = int((event.x - x_offset) / scale)
            img_y = int((event.y - y_offset) / scale)
            
            # clamp to image bounds
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))
            
            return img_x, img_y
            
        except Exception as e:
            print(f"coordinate conversion error: {e}")
            return None, None
            
    def combine_masks(self, existing_mask: Image.Image, new_mask: Image.Image) -> Image.Image:
        """combine two selection masks using OR operation"""
        try:
            if existing_mask is None:
                return new_mask
            if new_mask is None:
                return existing_mask
                
            # ensure both masks are the same size
            if existing_mask.size != new_mask.size:
                new_mask = new_mask.resize(existing_mask.size, Image.Resampling.LANCZOS)
            
            # convert to numpy arrays for efficient OR operation
            existing_array = np.array(existing_mask)
            new_array = np.array(new_mask)
            
            # combine using logical OR (white pixels from either mask)
            combined_array = np.maximum(existing_array, new_array)
            
            return Image.fromarray(combined_array, mode='L')
            
        except Exception as e:
            print(f"mask combination error: {e}")
            return existing_mask or new_mask
            
    def subtract_masks(self, existing_mask: Image.Image, subtract_mask: Image.Image) -> Image.Image:
        """subtract one selection mask from another"""
        try:
            if existing_mask is None or subtract_mask is None:
                return existing_mask
                
            # ensure both masks are the same size
            if existing_mask.size != subtract_mask.size:
                subtract_mask = subtract_mask.resize(existing_mask.size, Image.Resampling.LANCZOS)
            
            # convert to numpy arrays
            existing_array = np.array(existing_mask)
            subtract_array = np.array(subtract_mask)
            
            # subtract: where subtract_mask is white, set existing to black
            result_array = existing_array.copy()
            result_array[subtract_array > 128] = 0  # where subtract is white, result becomes black
            
            return Image.fromarray(result_array, mode='L')
            
        except Exception as e:
            print(f"mask subtraction error: {e}")
            return existing_mask

    def on_image_click(self, event):
        """handle mouse click on image"""
        if self.selection_mode == 'none':
            return
            
        try:
            # check for modifier keys
            is_additive = (event.state & 0x8) != 0 or (event.state & 0x4) != 0  # cmd or ctrl  
            is_subtractive = (event.state & 0x20000) != 0 or (event.state & 0x1) != 0  # alt or shift
            
            # determine which image was clicked
            is_front = (event.widget == self.image_label)
            img_x, img_y = self.get_image_coordinates(event, is_front)
            
            if img_x is None or img_y is None:
                return
                
            modifier_text = ""
            if is_additive:
                modifier_text = " (additive)"
            elif is_subtractive:
                modifier_text = " (subtractive)"
            print(f"image click: {img_x}, {img_y} ({'front' if is_front else 'back'}){modifier_text}")
            
            if self.selection_mode == 'color':
                # magic wand color selection
                if is_front and self.original_image:
                    new_mask = self.flood_fill_selection(self.original_image, img_x, img_y, self.color_tolerance)
                    
                    if is_additive and self.selection_mask:
                        # combine with existing selection
                        self.selection_mask = self.combine_masks(self.selection_mask, new_mask)
                    elif is_subtractive and self.selection_mask:
                        # subtract from existing selection
                        self.selection_mask = self.subtract_masks(self.selection_mask, new_mask)
                    else:
                        # replace existing selection
                        self.selection_mask = new_mask
                        
                elif not is_front and self.original_image_back:
                    new_mask = self.flood_fill_selection(self.original_image_back, img_x, img_y, self.color_tolerance)
                    
                    if is_additive and self.selection_mask_back:
                        # combine with existing selection
                        self.selection_mask_back = self.combine_masks(self.selection_mask_back, new_mask)
                    elif is_subtractive and self.selection_mask_back:
                        # subtract from existing selection
                        self.selection_mask_back = self.subtract_masks(self.selection_mask_back, new_mask)
                    else:
                        # replace existing selection
                        self.selection_mask_back = new_mask
                    
                # refresh display immediately
                self.refresh_image_display()
                
            elif self.selection_mode == 'brush':
                # brush selection - determine mode for this drawing session
                if is_additive:
                    self.current_draw_mode = 'add'
                elif is_subtractive:
                    self.current_draw_mode = 'subtract'
                else:
                    self.current_draw_mode = self.brush_mode  # use current brush mode setting
                
                self.is_drawing = True
                self.last_draw_pos = (img_x, img_y)
                self.apply_selection_to_mask(img_x, img_y, is_front)
                self.refresh_image_display()
                
        except Exception as e:
            print(f"image click error: {e}")
            
    def on_image_drag(self, event):
        """handle mouse drag on image"""
        if self.selection_mode != 'brush' or not self.is_drawing:
            return
            
        try:
            # determine which image
            is_front = (event.widget == self.image_label)
            img_x, img_y = self.get_image_coordinates(event, is_front)
            
            if img_x is None or img_y is None:
                return
                
            # continue brush stroke (brush mode was set during click event)
            self.apply_selection_to_mask(img_x, img_y, is_front)
            self.refresh_image_display()
            
        except Exception as e:
            print(f"image drag error: {e}")
            
    def on_image_release(self, event):
        """handle mouse release on image"""
        if self.selection_mode == 'brush':
            self.is_drawing = False
            self.last_draw_pos = None
        
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
            print(f"initializing gui for card: {card_name}")
            self.printings = printings
            self.current_index = 0
            self.selected_printing = None
            
            # reset per-dialog loading state (but keep global cache!)
            self.loading_status.clear()
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            self.executor = ThreadPoolExecutor(max_workers=3)
            
            print("creating tkinter window...")
            # create main window - wider to accommodate double-sided cards + adjustment controls
            self.root = tk.Tk()
            self.root.title(f"Select Card Art - {card_name}")
            self.root.geometry("1000x800")  # wider for adjustment controls
            self.root.resizable(True, True)
            
            # restore window position if we have one saved, but validate it's on-screen
            if self.window_x is not None and self.window_y is not None:
                # make sure the window position is reasonable (not off-screen)
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                
                # clamp position to screen bounds
                safe_x = max(0, min(self.window_x, screen_width - 1000))
                safe_y = max(0, min(self.window_y, screen_height - 800))
                
                print(f"restoring window position: {safe_x},{safe_y}")
                self.root.geometry(f"1000x800+{safe_x}+{safe_y}")
            else:
                # center window on screen
                screen_width = self.root.winfo_screenwidth()
                screen_height = self.root.winfo_screenheight()
                center_x = (screen_width - 1000) // 2
                center_y = (screen_height - 800) // 2
                print(f"centering window at: {center_x},{center_y}")
                self.root.geometry(f"1000x800+{center_x}+{center_y}")
            
            # force window to front and focus
            self.root.lift()
            self.root.attributes('-topmost', True)
            self.root.after_idle(lambda: self.root.attributes('-topmost', False))
            self.root.focus_force()
            
            print("creating gui elements...")
            # main container with two columns: left for card display, right for controls
            main_container = ttk.Frame(self.root, padding="10")
            main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            main_container.columnconfigure(0, weight=2)  # card area gets more space
            main_container.columnconfigure(1, weight=1)  # controls area
            main_container.rowconfigure(0, weight=1)
            
            # left frame for card display
            card_frame = ttk.Frame(main_container)
            card_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
            
            # right frame for adjustment controls  
            controls_frame = ttk.LabelFrame(main_container, text="Image Adjustments", padding="10")
            controls_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            # === CARD DISPLAY AREA ===
            # card name label
            name_label = ttk.Label(card_frame, text=card_name, font=("Arial", 16, "bold"))
            name_label.grid(row=0, column=0, columnspan=3, pady=(0, 10))
            
            # counter label
            self.counter_label = ttk.Label(card_frame, text="", font=("Arial", 12))
            self.counter_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))
            
            # create image container frame to support side-by-side layout
            self.image_frame = ttk.Frame(card_frame)
            self.image_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10))
            self.image_frame.columnconfigure(0, weight=1)
            self.image_frame.columnconfigure(1, weight=1)
            
            # image display (front image goes in column 0)
            self.image_label = ttk.Label(self.image_frame, text="loading image...", anchor="center")
            self.image_label.grid(row=2, column=0, pady=(0, 10))
            
            # bind mouse events for selection
            self.image_label.bind("<Button-1>", self.on_image_click)
            self.image_label.bind("<B1-Motion>", self.on_image_drag)
            self.image_label.bind("<ButtonRelease-1>", self.on_image_release)
            
            # navigation buttons frame
            nav_frame = ttk.Frame(card_frame)
            nav_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10))
            
            prev_button = ttk.Button(nav_frame, text="◀ Previous", command=self.previous_card)
            prev_button.pack(side=tk.LEFT, padx=(0, 10))
            
            next_button = ttk.Button(nav_frame, text="Next ▶", command=self.next_card)
            next_button.pack(side=tk.LEFT)
            
            # card info label
            self.info_label = ttk.Label(card_frame, text="", justify=tk.LEFT, font=("Arial", 10))
            self.info_label.grid(row=4, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
            
            # action buttons frame
            button_frame = ttk.Frame(card_frame)
            button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
            
            select_button = ttk.Button(button_frame, text="Select This Version", command=self.select_card)
            select_button.pack(side=tk.LEFT, padx=(0, 10))
            
            skip_button = ttk.Button(button_frame, text="Use First Printing", command=self.skip_card)
            skip_button.pack(side=tk.LEFT)
            
            # === ADJUSTMENT CONTROLS ===
            self.create_adjustment_controls(controls_frame)
            
            # keyboard bindings
            self.root.bind('<Left>', lambda e: self.previous_card())
            self.root.bind('<Right>', lambda e: self.next_card())
            self.root.bind('<Return>', lambda e: self.select_card())
            self.root.bind('<Escape>', lambda e: self.skip_card())
            
            # configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            card_frame.columnconfigure(0, weight=1)
            
            # initial display and start background loading
            if self.printings:
                print("initializing image loading...")
                
                if self.use_caching:
                    # advanced mode with caching
                    try:
                        print("attempting advanced loading with caching...")
                        # show cache statistics
                        with self._cache_lock:
                            cache_size_mb = self._estimate_cache_size_mb()
                            cached_count = len(self._global_image_cache)
                        
                        print(f"starting background loading for {len(self.printings)} printings... (cache: {cached_count} images, {cache_size_mb:.1f}mb)")
                        
                        # start background loading - this will load first printing immediately
                        self.start_background_loading()
                        
                        print("background loading started, updating display...")
                        # update display (will show first printing when loaded)
                        self.update_display()
                        
                        print("starting refresh scheduler...")
                        # start periodic refresh to update loading progress
                        self.schedule_refresh()
                        
                        print("advanced loading setup complete")
                        
                    except Exception as e:
                        print(f"error during advanced loading setup: {e}")
                        print("falling back to simple mode...")
                        self.use_caching = False  # disable caching for this session
                        
                if not self.use_caching:
                    # simple mode - just show first printing without caching
                    try:
                        if self.simple_cache:
                            cache_count = len(self.simple_image_cache)
                            print(f"using simple mode with caching... (cache: {cache_count} images)")
                        else:
                            print("using simple mode (no caching)...")
                        printing = self.printings[0]
                        self.display_simple_first_card(printing)
                        print("simple mode setup complete")
                        
                    except Exception as e:
                        print(f"error in simple mode: {e}")
                        self.display_loading_message("Ready - use arrow keys to navigate")
                
            print("showing gui window...")
            # force window to be visible
            self.root.deiconify()  # make sure window is not minimized
            self.root.update()     # force update before mainloop
            
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