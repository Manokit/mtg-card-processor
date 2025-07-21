import tkinter as tk
from tkinter import ttk, messagebox
import requests
from PIL import Image, ImageTk, ImageEnhance, ImageOps
from io import BytesIO
import time
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
import os
import pickle
import hashlib

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
        
    def apply_image_adjustments(self, image: Image.Image) -> Image.Image:
        """apply all current adjustments to an image"""
        if not image:
            return None
            
        try:
            # start with copy of original and ensure RGB mode
            adjusted = image.copy()
            if adjusted.mode != 'RGB':
                adjusted = adjusted.convert('RGB')
            
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
            
            return adjusted
            
        except Exception as e:
            print(f"error applying image adjustments: {e}")
            return image  # return original if adjustment fails

    def display_single_card_simple(self, image_url: str):
        """display a single card image with optional simple caching and adjustments"""
        try:
            image = self.get_image_with_simple_cache(image_url)
            
            # store original for adjustments
            self.original_image = image.copy()
            self.original_image_back = None
            
            # apply adjustments
            adjusted_image = self.apply_image_adjustments(image)
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
            
            # apply adjustments and resize
            adjusted_front = self.apply_image_adjustments(front_image)
            adjusted_front.thumbnail((250, 350), Image.Resampling.LANCZOS)
            self.photo_image = ImageTk.PhotoImage(adjusted_front)
            
            adjusted_back = self.apply_image_adjustments(back_image)
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
        """refresh the image display with current adjustments"""
        try:
            if self.original_image and self.original_image_back:
                # double-sided card
                adjusted_front = self.apply_image_adjustments(self.original_image)
                adjusted_front.thumbnail((250, 350), Image.Resampling.LANCZOS)
                self.photo_image = ImageTk.PhotoImage(adjusted_front)
                
                adjusted_back = self.apply_image_adjustments(self.original_image_back)  
                adjusted_back.thumbnail((250, 350), Image.Resampling.LANCZOS)
                self.photo_image_back = ImageTk.PhotoImage(adjusted_back)
                
                # update display
                self.image_label.configure(image=self.photo_image)
                if hasattr(self, 'image_label_back'):
                    self.image_label_back.configure(image=self.photo_image_back)
                    
            elif self.original_image:
                # single card
                adjusted_image = self.apply_image_adjustments(self.original_image)
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
            
            # also store the adjusted images for saving
            if self.original_image and self.original_image_back:
                # double-sided card - store both adjusted images
                self.selected_printing['adjusted_front_image'] = self.apply_image_adjustments(self.original_image)
                self.selected_printing['adjusted_back_image'] = self.apply_image_adjustments(self.original_image_back)
            elif self.original_image:
                # single card - store adjusted image
                self.selected_printing['adjusted_image'] = self.apply_image_adjustments(self.original_image)
                
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
        
        # brightness control
        ttk.Label(parent, text="Brightness:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        self.brightness_var = tk.DoubleVar(value=self.brightness)
        brightness_scale = ttk.Scale(parent, from_=0.3, to=2.0, variable=self.brightness_var, 
                                   orient=tk.HORIZONTAL, length=180,
                                   command=self.on_brightness_change)
        brightness_scale.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.brightness_value_label = ttk.Label(parent, text=f"{self.brightness:.2f}")
        self.brightness_value_label.grid(row=row, column=1, padx=(10, 0), pady=(0, 10))
        row += 1
        
        # contrast control  
        ttk.Label(parent, text="Contrast:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        self.contrast_var = tk.DoubleVar(value=self.contrast)
        contrast_scale = ttk.Scale(parent, from_=0.3, to=2.5, variable=self.contrast_var, 
                                 orient=tk.HORIZONTAL, length=180,
                                 command=self.on_contrast_change)
        contrast_scale.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.contrast_value_label = ttk.Label(parent, text=f"{self.contrast:.2f}")
        self.contrast_value_label.grid(row=row, column=1, padx=(10, 0), pady=(0, 10))
        row += 1
        
        # saturation control
        ttk.Label(parent, text="Saturation:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        self.saturation_var = tk.DoubleVar(value=self.saturation)
        saturation_scale = ttk.Scale(parent, from_=0.0, to=2.0, variable=self.saturation_var, 
                                   orient=tk.HORIZONTAL, length=180,
                                   command=self.on_saturation_change)
        saturation_scale.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.saturation_value_label = ttk.Label(parent, text=f"{self.saturation:.2f}")
        self.saturation_value_label.grid(row=row, column=1, padx=(10, 0), pady=(0, 10))
        row += 1
        
        # gamma control
        ttk.Label(parent, text="Gamma:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        self.gamma_var = tk.DoubleVar(value=self.gamma)
        gamma_scale = ttk.Scale(parent, from_=0.3, to=2.5, variable=self.gamma_var, 
                              orient=tk.HORIZONTAL, length=180,
                              command=self.on_gamma_change)
        gamma_scale.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.gamma_value_label = ttk.Label(parent, text=f"{self.gamma:.2f}")
        self.gamma_value_label.grid(row=row, column=1, padx=(10, 0), pady=(0, 10))
        row += 1
        
        # color balance control
        ttk.Label(parent, text="Color Temperature:", font=("Arial", 10, "bold")).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        ttk.Label(parent, text="(Cool ← → Warm)", font=("Arial", 8)).grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        row += 1
        self.color_balance_var = tk.DoubleVar(value=self.color_balance)
        color_balance_scale = ttk.Scale(parent, from_=-100, to=100, variable=self.color_balance_var, 
                                      orient=tk.HORIZONTAL, length=180,
                                      command=self.on_color_balance_change)
        color_balance_scale.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.color_balance_value_label = ttk.Label(parent, text=f"{self.color_balance:.0f}")
        self.color_balance_value_label.grid(row=row, column=1, padx=(10, 0), pady=(0, 10))
        row += 1
        
        # separator
        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=20)
        row += 1
        
        # reset button
        reset_button = ttk.Button(parent, text="Reset All", command=self.reset_adjustments)
        reset_button.grid(row=row, column=0, columnspan=2, pady=(0, 10))
        row += 1
        
        # tips label
        tips_text = "Tips for MTG cards:\n• Increase contrast for borders\n• Adjust gamma for text areas\n• Warm colors for older sets\n• Cool colors for modern sets"
        tips_label = ttk.Label(parent, text=tips_text, font=("Arial", 9), justify=tk.LEFT, 
                              foreground="gray", wraplength=200)
        tips_label.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 0))
        
        # configure column weights  
        parent.columnconfigure(0, weight=1)
        
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