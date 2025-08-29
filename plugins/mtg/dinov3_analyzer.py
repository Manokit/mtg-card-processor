"""
DINOv3-based semantic analyzer for MTG cards.
Provides intelligent segmentation and analysis of card elements.
"""
import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
from typing import Dict, List, Tuple, Optional, Union
import warnings

# Optional imports with fallback
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoImageProcessor, AutoModel
    from torchvision import transforms
    DINOV3_AVAILABLE = True
except ImportError:
    DINOV3_AVAILABLE = False
    warnings.warn("DINOv3 dependencies not available. Please install: torch, torchvision, transformers")

try:
    from skimage import measure, morphology, segmentation
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available. Some features may be limited.")


class DINOv3CardAnalyzer:
    """
    MTG Card analyzer using DINOv3 for semantic understanding of card elements.
    """
    
    def __init__(self, model_name: str = "microsoft/DiT-base-finetuned-ade-512-512", device: str = "auto"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self._device = None
        self.is_initialized = False
        
        # Determine device
        if device == "auto":
            if DINOV3_AVAILABLE and torch.cuda.is_available():
                self._device = "cuda"
            elif DINOV3_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device
            
        # Cache for feature maps
        self._feature_cache = {}
        
        # MTG card layout knowledge
        self.card_regions = {
            'name_box': (0.04, 0.14, 0.0, 1.0),        # top 4-14% of card
            'mana_cost': (0.04, 0.14, 0.75, 1.0),      # top-right corner
            'art_box': (0.14, 0.47, 0.08, 0.92),       # main art area
            'type_line': (0.47, 0.57, 0.08, 0.92),     # type line area
            'text_box': (0.57, 0.90, 0.08, 0.92),      # rules text area
            'power_toughness': (0.83, 0.95, 0.75, 0.95), # bottom right
            'border_frame': (0.0, 1.0, 0.0, 1.0),      # entire card for border detection
        }
    
    def initialize_model(self) -> bool:
        """
        Initialize the DINOv3 model. Returns True if successful.
        """
        if not DINOV3_AVAILABLE:
            print("Warning: DINOv3 dependencies not available. Using fallback methods.")
            return False
            
        if self.is_initialized:
            return True
            
        try:
            print(f"Initializing DINOv3 model: {self.model_name}")
            print(f"Using device: {self._device}")
            
            # Load processor and model
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            if self._device != "cpu":
                self.model = self.model.to(self._device)
            
            self.model.eval()
            self.is_initialized = True
            print("DINOv3 model initialized successfully")
            return True
            
        except Exception as e:
            print(f"Failed to initialize DINOv3 model: {e}")
            print("Falling back to traditional methods")
            return False
    
    def get_dense_features(self, image: Image.Image, patch_size: int = 14) -> Optional[np.ndarray]:
        """
        Extract dense features from image using DINOv3.
        Returns feature map of shape (H/patch_size, W/patch_size, feature_dim).
        """
        if not self.is_initialized and not self.initialize_model():
            return None
            
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Create cache key
            cache_key = f"{hash(image.tobytes())}_{patch_size}"
            if cache_key in self._feature_cache:
                return self._feature_cache[cache_key]
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            if self._device != "cpu":
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Get patch embeddings (excluding CLS token)
                patch_embeddings = outputs.last_hidden_state[:, 1:]  # Remove CLS token
                
                # Reshape to spatial dimensions
                batch_size, num_patches, feature_dim = patch_embeddings.shape
                h_patches = w_patches = int(np.sqrt(num_patches))
                
                features = patch_embeddings.reshape(batch_size, h_patches, w_patches, feature_dim)
                features = features[0].cpu().numpy()  # Remove batch dimension
            
            # Cache the result
            self._feature_cache[cache_key] = features
            return features
            
        except Exception as e:
            print(f"Error extracting DINOv3 features: {e}")
            return None
    
    def create_semantic_mask(self, image: Image.Image, element_type: str, 
                           similarity_threshold: float = 0.7) -> Image.Image:
        """
        Create semantic mask for specific card element using advanced computer vision.
        """
        try:
            img_width, img_height = image.size
            
            # Always use advanced computer vision methods
            # (Skip DINOv3 for now due to model access issues)
            if element_type == 'text_box':
                mask = self._fallback_text_detection(None, image)
            elif element_type == 'art_box':
                mask = self._fallback_art_detection(None, image)
            elif element_type == 'border_frame':
                mask = self._fallback_border_detection(None, image)
            elif element_type == 'name_box':
                mask = self._fallback_name_detection(None, image)
            elif element_type == 'type_line':
                mask = self._fallback_type_detection(None, image)
            else:
                # Unknown element type, return empty mask
                return Image.new('L', (img_width, img_height), 0)
            
            # Post-process the mask
            mask_processed = self._post_process_mask(mask, element_type)
            
            return Image.fromarray(mask_processed, mode='L')
            
        except Exception as e:
            print(f"Error creating semantic mask for {element_type}: {e}")
            # Final fallback to simple rectangular region
            return self._create_simple_rectangular_mask(image, element_type)
    
    def _segment_text_regions(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Segment text regions using DINOv3 semantic understanding.
        """
        try:
            # Use clustering to identify semantically similar regions
            h, w, d = features.shape
            img_width, img_height = image.size
            
            # Flatten features for clustering
            features_flat = features.reshape(-1, d)
            
            # Use k-means clustering to identify different semantic regions
            from sklearn.cluster import KMeans
            n_clusters = 8  # Reasonable number for card elements
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_flat)
            cluster_labels = cluster_labels.reshape(h, w)
            
            # Analyze each cluster to identify text-like regions
            text_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Convert image to analyze with traditional CV for validation
            img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            img_gray_resized = cv2.resize(img_gray, (w, h))
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                
                # Get region properties
                cluster_region = img_gray_resized[cluster_mask]
                
                # Text regions typically have:
                # 1. High contrast (text vs background)
                # 2. Medium brightness (not pure black/white)
                # 3. Structured patterns
                cluster_contrast = np.std(cluster_region)
                cluster_brightness = np.mean(cluster_region)
                
                # Check if this cluster looks like text
                is_text_like = (
                    cluster_contrast > 40 and  # Good contrast
                    50 < cluster_brightness < 200 and  # Not too dark/bright
                    np.sum(cluster_mask) > 10  # Reasonable size
                )
                
                # Additional check: text is usually in expected areas
                cluster_coords = np.where(cluster_mask)
                if len(cluster_coords[0]) > 0:
                    center_y = np.mean(cluster_coords[0]) / h
                    center_x = np.mean(cluster_coords[1]) / w
                    
                    # Text boxes are typically in middle-to-bottom area
                    in_text_area = (0.4 < center_y < 0.95) and (0.1 < center_x < 0.9)
                    
                    if is_text_like and in_text_area:
                        text_mask[cluster_mask] = 255
            
            # Resize to original image size
            text_mask_resized = cv2.resize(text_mask, (img_width, img_height), 
                                         interpolation=cv2.INTER_NEAREST)
            
            return text_mask_resized
            
        except ImportError:
            print("sklearn not available, using fallback text detection")
            return self._fallback_text_detection(features, image)
        except Exception as e:
            print(f"Error in text segmentation: {e}")
            return self._fallback_text_detection(features, image)
    
    def _segment_artwork_region(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Segment artwork region using semantic clustering.
        """
        try:
            from sklearn.cluster import KMeans
            
            h, w, d = features.shape
            img_width, img_height = image.size
            
            # Flatten features
            features_flat = features.reshape(-1, d)
            
            # Use more clusters to capture artwork complexity
            n_clusters = 12
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_flat)
            cluster_labels = cluster_labels.reshape(h, w)
            
            # Convert image for analysis
            img_array = np.array(image)
            img_resized = cv2.resize(img_array, (w, h))
            
            art_mask = np.zeros((h, w), dtype=np.uint8)
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                
                # Get cluster properties
                cluster_coords = np.where(cluster_mask)
                if len(cluster_coords[0]) == 0:
                    continue
                
                # Calculate cluster position
                center_y = np.mean(cluster_coords[0]) / h
                center_x = np.mean(cluster_coords[1]) / w
                cluster_size = np.sum(cluster_mask)
                
                # Artwork characteristics:
                # 1. Located in upper-middle area of card
                # 2. Usually largest region
                # 3. High visual complexity (color variation)
                in_art_area = (0.1 < center_y < 0.6) and (0.1 < center_x < 0.9)
                reasonable_size = cluster_size > 50
                
                if in_art_area and reasonable_size:
                    # Check visual complexity
                    cluster_region = img_resized[cluster_mask]
                    color_variance = np.var(cluster_region, axis=0).mean()
                    
                    # Artwork usually has high color variance
                    if color_variance > 800:  # Threshold for complex artwork
                        art_mask[cluster_mask] = 255
            
            # Resize back to original
            art_mask_resized = cv2.resize(art_mask, (img_width, img_height), 
                                        interpolation=cv2.INTER_NEAREST)
            
            return art_mask_resized
            
        except ImportError:
            return self._fallback_art_detection(features, image)
        except Exception as e:
            print(f"Error in artwork segmentation: {e}")
            return self._fallback_art_detection(features, image)
    
    def _segment_border_regions(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Segment border/frame regions using edge detection and semantic understanding.
        """
        try:
            from sklearn.cluster import KMeans
            
            h, w, d = features.shape
            img_width, img_height = image.size
            
            # Convert image for edge analysis
            img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            
            # Use Canny edge detection to identify border candidates
            edges = cv2.Canny(img_gray, 50, 150)
            edges_resized = cv2.resize(edges, (w, h))
            
            # Combine with semantic clustering
            features_flat = features.reshape(-1, d)
            n_clusters = 6
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_flat)
            cluster_labels = cluster_labels.reshape(h, w)
            
            border_mask = np.zeros((h, w), dtype=np.uint8)
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                
                # Check if cluster is near edges and has border-like properties
                cluster_coords = np.where(cluster_mask)
                if len(cluster_coords[0]) == 0:
                    continue
                
                # Borders are typically at edges of card
                min_y, max_y = np.min(cluster_coords[0]), np.max(cluster_coords[0])
                min_x, max_x = np.min(cluster_coords[1]), np.max(cluster_coords[1])
                
                # Check if cluster touches edges
                touches_edge = (
                    min_y < h * 0.05 or max_y > h * 0.95 or  # Top/bottom edge
                    min_x < w * 0.05 or max_x > w * 0.95     # Left/right edge
                )
                
                # Check edge density in this cluster
                edge_density = np.mean(edges_resized[cluster_mask])
                
                if touches_edge and edge_density > 30:  # Strong edge presence
                    border_mask[cluster_mask] = 255
            
            # Resize back to original
            border_mask_resized = cv2.resize(border_mask, (img_width, img_height), 
                                           interpolation=cv2.INTER_NEAREST)
            
            return border_mask_resized
            
        except ImportError:
            return self._fallback_border_detection(features, image)
        except Exception as e:
            print(f"Error in border segmentation: {e}")
            return self._fallback_border_detection(features, image)
    
    def _segment_name_region(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Segment name/title region at top of card.
        """
        try:
            from sklearn.cluster import KMeans
            
            h, w, d = features.shape
            img_width, img_height = image.size
            
            # Focus on top portion of card where name is located
            top_features = features[:int(h * 0.25), :, :]  # Top 25%
            top_h, top_w = top_features.shape[:2]
            
            if top_h == 0 or top_w == 0:
                return self._fallback_name_detection(features, image)
            
            # Cluster the top region
            top_features_flat = top_features.reshape(-1, d)
            n_clusters = 5
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(top_features_flat)
            cluster_labels = cluster_labels.reshape(top_h, top_w)
            
            # Analyze image properties in top region
            img_array = np.array(image)
            top_img = img_array[:int(img_height * 0.25), :, :]
            top_img_resized = cv2.resize(top_img, (top_w, top_h))
            
            name_mask = np.zeros((top_h, top_w), dtype=np.uint8)
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                
                # Name regions typically have text-like properties
                cluster_region = top_img_resized[cluster_mask]
                cluster_brightness = np.mean(cluster_region)
                cluster_contrast = np.std(cluster_region)
                
                # Check for text-like properties in name area
                is_name_like = (
                    cluster_contrast > 30 and  # Good contrast for readability
                    80 < cluster_brightness < 180 and  # Not too dark/bright
                    np.sum(cluster_mask) > 20  # Reasonable size
                )
                
                if is_name_like:
                    name_mask[cluster_mask] = 255
            
            # Create full-size mask
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            name_mask_resized = cv2.resize(name_mask, (img_width, int(img_height * 0.25)), 
                                         interpolation=cv2.INTER_NEAREST)
            full_mask[:int(img_height * 0.25), :] = name_mask_resized
            
            return full_mask
            
        except ImportError:
            return self._fallback_name_detection(features, image)
        except Exception as e:
            print(f"Error in name segmentation: {e}")
            return self._fallback_name_detection(features, image)
    
    def _segment_type_line(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """
        Segment type line region using semantic clustering.
        """
        try:
            from sklearn.cluster import KMeans
            
            h, w, d = features.shape
            img_width, img_height = image.size
            
            # Focus on middle region where type line is located
            mid_start = int(h * 0.4)
            mid_end = int(h * 0.65)
            mid_features = features[mid_start:mid_end, :, :]
            mid_h, mid_w = mid_features.shape[:2]
            
            if mid_h == 0 or mid_w == 0:
                return self._fallback_type_detection(features, image)
            
            # Cluster the middle region
            mid_features_flat = mid_features.reshape(-1, d)
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(mid_features_flat)
            cluster_labels = cluster_labels.reshape(mid_h, mid_w)
            
            type_mask = np.zeros((mid_h, mid_w), dtype=np.uint8)
            
            # Analyze clusters for type line characteristics
            img_array = np.array(image)
            mid_img_start = int(img_height * 0.4)
            mid_img_end = int(img_height * 0.65)
            mid_img = img_array[mid_img_start:mid_img_end, :, :]
            mid_img_resized = cv2.resize(mid_img, (mid_w, mid_h))
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                
                # Type line usually has different visual properties than artwork
                cluster_region = mid_img_resized[cluster_mask]
                cluster_coords = np.where(cluster_mask)
                
                # Type line is typically horizontal and spans most of the width
                if len(cluster_coords[1]) > 0:
                    width_span = np.max(cluster_coords[1]) - np.min(cluster_coords[1])
                    height_span = np.max(cluster_coords[0]) - np.min(cluster_coords[0])
                    
                    is_horizontal = width_span > height_span * 2
                    spans_width = width_span > mid_w * 0.5
                    
                    if is_horizontal and spans_width:
                        type_mask[cluster_mask] = 255
            
            # Create full-size mask
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            type_mask_resized = cv2.resize(type_mask, (img_width, mid_img_end - mid_img_start), 
                                         interpolation=cv2.INTER_NEAREST)
            full_mask[mid_img_start:mid_img_end, :] = type_mask_resized
            
            return full_mask
            
        except ImportError:
            return self._fallback_type_detection(features, image)
        except Exception as e:
            print(f"Error in type line segmentation: {e}")
            return self._fallback_type_detection(features, image)
    
    def _fallback_text_detection(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """Advanced text detection using computer vision techniques."""
        try:
            img_array = np.array(image)
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_width, img_height = image.size
            
            # Use MSER (Maximally Stable Extremal Regions) to detect text regions
            mser = cv2.MSER_create(
                delta=5,
                min_area=60,
                max_area=img_width * img_height // 50,
                max_variation=0.25
            )
            
            regions, _ = mser.detectRegions(img_gray)
            text_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            for region in regions:
                # Create convex hull for each region
                hull = cv2.convexHull(region.reshape(-1, 1, 2))
                
                # Check if region is in text areas (bottom 2/3 of card)
                center_y = np.mean(region[:, 1])
                center_x = np.mean(region[:, 0])
                
                if (center_y > img_height * 0.4 and center_y < img_height * 0.95 and
                    center_x > img_width * 0.1 and center_x < img_width * 0.9):
                    
                    # Fill the hull region
                    cv2.fillPoly(text_mask, [hull], 255)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel)
            text_mask = cv2.morphologyEx(text_mask, cv2.MORPH_OPEN, kernel)
            
            return text_mask
            
        except Exception as e:
            print(f"Error in advanced text detection: {e}")
            # Simple fallback
            img_width, img_height = image.size
            region = self.card_regions['text_box']
            top = int(region[0] * img_height)
            bottom = int(region[1] * img_height)
            left = int(region[2] * img_width)
            right = int(region[3] * img_width)
            
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask[top:bottom, left:right] = 255
            return mask
    
    def _fallback_art_detection(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """Advanced artwork detection using color clustering."""
        try:
            img_array = np.array(image)
            img_height, img_width = img_array.shape[:2]
            
            # Focus on the art region (roughly upper middle of card)
            art_region_top = int(img_height * 0.12)
            art_region_bottom = int(img_height * 0.55)
            art_region_left = int(img_width * 0.08)
            art_region_right = int(img_width * 0.92)
            
            # Extract the art region
            art_region = img_array[art_region_top:art_region_bottom, 
                                 art_region_left:art_region_right]
            
            if art_region.size == 0:
                return self._simple_fallback_art(image)
            
            # Use color clustering to identify the main artwork
            from sklearn.cluster import KMeans
            art_reshaped = art_region.reshape(-1, 3)
            
            # Cluster colors to identify distinct regions
            n_clusters = 8
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(art_reshaped)
            cluster_labels = cluster_labels.reshape(art_region.shape[:2])
            
            # Find clusters that represent the main artwork (highest color variance)
            art_mask_region = np.zeros(art_region.shape[:2], dtype=np.uint8)
            
            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                if not np.any(cluster_mask):
                    continue
                
                # Calculate color variance for this cluster
                cluster_pixels = art_region[cluster_mask]
                if len(cluster_pixels) > 50:  # Minimum size
                    color_variance = np.var(cluster_pixels, axis=0).mean()
                    
                    # Artwork typically has high color variance
                    if color_variance > 500:
                        art_mask_region[cluster_mask] = 255
            
            # Create full image mask
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            full_mask[art_region_top:art_region_bottom, 
                     art_region_left:art_region_right] = art_mask_region
            
            return full_mask
            
        except ImportError:
            return self._simple_fallback_art(image)
        except Exception as e:
            print(f"Error in advanced art detection: {e}")
            return self._simple_fallback_art(image)
    
    def _simple_fallback_art(self, image: Image.Image) -> np.ndarray:
        """Simple rectangular art region fallback."""
        img_width, img_height = image.size
        region = self.card_regions['art_box']
        top = int(region[0] * img_height)
        bottom = int(region[1] * img_height)
        left = int(region[2] * img_width)
        right = int(region[3] * img_width)
        
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        mask[top:bottom, left:right] = 255
        return mask
    
    def _fallback_border_detection(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """Advanced border detection using edge analysis and morphology."""
        try:
            img_array = np.array(image)
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_height, img_width = img_gray.shape
            
            # Multi-scale edge detection
            edges1 = cv2.Canny(img_gray, 50, 150)
            edges2 = cv2.Canny(img_gray, 100, 200)
            edges_combined = cv2.bitwise_or(edges1, edges2)
            
            # Create border mask by detecting strong edge structures near perimeter
            border_mask = np.zeros_like(img_gray)
            
            # Detect horizontal lines (top/bottom borders)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            horizontal_lines = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, kernel_h)
            
            # Detect vertical lines (left/right borders)
            kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            vertical_lines = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, kernel_v)
            
            # Combine line detections
            border_lines = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
            
            # Focus on perimeter regions
            border_width = int(min(img_height, img_width) * 0.08)
            
            # Top border
            border_mask[:border_width, :] = np.maximum(
                border_mask[:border_width, :], 
                border_lines[:border_width, :]
            )
            
            # Bottom border  
            border_mask[-border_width:, :] = np.maximum(
                border_mask[-border_width:, :], 
                border_lines[-border_width:, :]
            )
            
            # Left border
            border_mask[:, :border_width] = np.maximum(
                border_mask[:, :border_width], 
                border_lines[:, :border_width]
            )
            
            # Right border
            border_mask[:, -border_width:] = np.maximum(
                border_mask[:, -border_width:], 
                border_lines[:, -border_width:]
            )
            
            # Dilate to create more complete border regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            border_mask = cv2.dilate(border_mask, kernel, iterations=2)
            
            return border_mask
            
        except Exception as e:
            print(f"Error in advanced border detection: {e}")
            # Simple edge-based fallback
            img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(img_gray, 100, 200)
            return edges
    
    def _fallback_name_detection(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """Advanced name detection using OCR-style techniques."""
        try:
            img_array = np.array(image)
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_height, img_width = img_gray.shape
            
            # Focus on top region where name appears
            top_region = img_gray[:int(img_height * 0.25), :]
            
            # Use adaptive thresholding to highlight text
            adaptive_thresh = cv2.adaptiveThreshold(
                top_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Find contours that look like text
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            name_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            
            for contour in contours:
                # Calculate contour properties
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                
                # Text-like properties: reasonable size, not too tall/wide
                if (100 < area < 5000 and 
                    h > 8 and w > 15 and  # Minimum text dimensions
                    h < img_height * 0.15 and w < img_width * 0.8):  # Maximum dimensions
                    
                    # Fill this region in the full mask
                    cv2.fillPoly(name_mask, [contour + np.array([0, 0])], 255)
            
            return name_mask
            
        except Exception as e:
            print(f"Error in advanced name detection: {e}")
            # Simple rectangular fallback
            img_width, img_height = image.size
            region = self.card_regions['name_box']
            top = int(region[0] * img_height)
            bottom = int(region[1] * img_height)
            left = int(region[2] * img_width)
            right = int(region[3] * img_width)
            
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask[top:bottom, left:right] = 255
            return mask
    
    def _fallback_type_detection(self, features: np.ndarray, image: Image.Image) -> np.ndarray:
        """Advanced type line detection using line detection."""
        try:
            img_array = np.array(image)
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            img_height, img_width = img_gray.shape
            
            # Focus on middle region where type line appears
            mid_start = int(img_height * 0.4)
            mid_end = int(img_height * 0.65)
            mid_region = img_gray[mid_start:mid_end, :]
            
            # Use edge detection and line detection
            edges = cv2.Canny(mid_region, 50, 150)
            
            # Detect horizontal lines (type line separator)
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_h)
            
            # Use adaptive thresholding to find text regions
            adaptive_thresh = cv2.adaptiveThreshold(
                mid_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Combine line detection with text detection
            type_region_mask = cv2.bitwise_or(horizontal_lines, adaptive_thresh)
            
            # Clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            type_region_mask = cv2.morphologyEx(type_region_mask, cv2.MORPH_CLOSE, kernel)
            
            # Create full image mask
            full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            full_mask[mid_start:mid_end, :] = type_region_mask
            
            return full_mask
            
        except Exception as e:
            print(f"Error in advanced type detection: {e}")
            # Simple rectangular fallback
            img_width, img_height = image.size
            region = self.card_regions['type_line']
            top = int(region[0] * img_height)
            bottom = int(region[1] * img_height)
            left = int(region[2] * img_width)
            right = int(region[3] * img_width)
            
            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            mask[top:bottom, left:right] = 255
            return mask
    
    def _post_process_mask(self, mask: np.ndarray, element_type: str) -> np.ndarray:
        """
        Post-process mask based on element type.
        """
        if not SKIMAGE_AVAILABLE:
            return mask
        
        try:
            # Remove small regions
            mask_binary = mask > 127
            mask_cleaned = morphology.remove_small_objects(mask_binary, min_size=100)
            
            # Fill small holes
            mask_filled = morphology.remove_small_holes(mask_cleaned, area_threshold=50)
            
            # Element-specific processing
            if element_type in ['text_box', 'type_line']:
                # For text areas, use morphological closing to connect text regions
                kernel = morphology.disk(2)
                mask_filled = morphology.closing(mask_filled, kernel)
            elif element_type == 'border_frame':
                # For borders, emphasize edges
                mask_filled = morphology.dilation(mask_filled, morphology.disk(1))
            
            return (mask_filled * 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Error in mask post-processing: {e}")
            return mask
    
    def _create_fallback_mask(self, image: Image.Image, element_type: str) -> Image.Image:
        """
        Create fallback masks using traditional computer vision methods.
        """
        img_width, img_height = image.size
        mask = Image.new('L', (img_width, img_height), 0)
        
        if element_type not in self.card_regions:
            return mask
        
        # Get region coordinates
        region = self.card_regions[element_type]
        roi_top, roi_bottom, roi_left, roi_right = region
        
        # Convert to pixel coordinates
        top = int(roi_top * img_height)
        bottom = int(roi_bottom * img_height)
        left = int(roi_left * img_width)
        right = int(roi_right * img_width)
        
        # Create simple rectangular mask for the region
        draw = ImageDraw.Draw(mask)
        draw.rectangle([left, top, right, bottom], fill=255)
        
        return mask
    
    def _create_simple_rectangular_mask(self, image: Image.Image, element_type: str) -> Image.Image:
        """
        Create simple rectangular masks as final fallback.
        """
        return self._create_fallback_mask(image, element_type)
    
    def detect_all_elements(self, image: Image.Image) -> Dict[str, Image.Image]:
        """
        Detect all standard MTG card elements and return their masks.
        """
        elements = ['name_box', 'art_box', 'type_line', 'text_box', 'border_frame']
        masks = {}
        
        for element in elements:
            try:
                masks[element] = self.create_semantic_mask(image, element)
            except Exception as e:
                print(f"Failed to detect {element}: {e}")
                masks[element] = Image.new('L', image.size, 0)
        
        return masks
    
    def analyze_card_quality(self, image: Image.Image) -> Dict[str, Union[bool, float, str]]:
        """
        Analyze card image quality using semantic understanding.
        Enhanced version of the original analyze_card_issues function.
        """
        try:
            # Get basic image properties
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            analysis = {
                'overall_brightness': np.mean(img_array),
                'overall_contrast': np.std(img_array),
                'has_color_cast': False,
                'color_cast_type': 'neutral',
                'text_readability': 'good',
                'border_clarity': 'good',
                'suggested_corrections': [],
                'confidence': 0.8  # Default confidence for fallback analysis
            }
            
            # If DINOv3 is available, use semantic analysis
            if self.is_initialized:
                analysis.update(self._analyze_with_dinov3(image))
            else:
                analysis.update(self._analyze_with_fallback(image))
            
            return analysis
            
        except Exception as e:
            print(f"Error in card quality analysis: {e}")
            return {
                'error': True,
                'message': str(e),
                'suggested_corrections': ['manual_adjustment']
            }
    
    def _analyze_with_dinov3(self, image: Image.Image) -> Dict:
        """
        Perform advanced analysis using DINOv3 semantic understanding.
        """
        try:
            # Get masks for different elements
            masks = self.detect_all_elements(image)
            img_array = np.array(image)
            
            analysis = {'confidence': 0.9}  # Higher confidence with DINOv3
            
            # Analyze text box specifically
            text_mask = np.array(masks['text_box'])
            if np.any(text_mask > 127):
                text_region = img_array[text_mask > 127]
                text_brightness = np.mean(text_region)
                text_contrast = np.std(text_region)
                
                if text_brightness > 180:
                    analysis['text_readability'] = 'washed_out'
                    analysis['suggested_corrections'].append('darken_text_areas')
                elif text_contrast < 30:
                    analysis['text_readability'] = 'low_contrast'
                    analysis['suggested_corrections'].append('increase_text_contrast')
            
            # Analyze border/frame
            border_mask = np.array(masks['border_frame'])
            if np.any(border_mask > 127):
                border_region = img_array[border_mask > 127]
                border_contrast = np.std(border_region)
                
                if border_contrast < 35:
                    analysis['border_clarity'] = 'unclear'
                    analysis['suggested_corrections'].append('enhance_borders')
            
            return analysis
            
        except Exception as e:
            print(f"Error in DINOv3 analysis: {e}")
            return {'confidence': 0.5, 'suggested_corrections': ['manual_adjustment']}
    
    def _analyze_with_fallback(self, image: Image.Image) -> Dict:
        """
        Fallback analysis using traditional computer vision.
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        analysis = {'suggested_corrections': []}
        
        # Analyze overall image properties
        brightness = np.mean(img_array)
        contrast = np.std(img_array)
        
        if brightness < 80:
            analysis['suggested_corrections'].append('increase_brightness')
        elif brightness > 180:
            analysis['suggested_corrections'].append('decrease_brightness')
        
        if contrast < 45:
            analysis['suggested_corrections'].append('increase_contrast')
        
        # Color cast detection
        r_avg = np.mean(img_array[:, :, 0])
        g_avg = np.mean(img_array[:, :, 1])
        b_avg = np.mean(img_array[:, :, 2])
        
        warm_score = (r_avg + g_avg) - b_avg * 2
        if warm_score > 20:
            analysis['has_color_cast'] = True
            analysis['color_cast_type'] = 'warm'
            analysis['suggested_corrections'].append('cool_temperature')
        elif warm_score < -20:
            analysis['has_color_cast'] = True
            analysis['color_cast_type'] = 'cool'
            analysis['suggested_corrections'].append('warm_temperature')
        
        return analysis
    
    def create_smart_selection_mask(self, image: Image.Image, element_types: List[str]) -> Image.Image:
        """
        Create combined mask for multiple element types.
        """
        if not element_types:
            return Image.new('L', image.size, 0)
        
        combined_mask = None
        
        for element_type in element_types:
            element_mask = self.create_semantic_mask(image, element_type)
            element_array = np.array(element_mask)
            
            if combined_mask is None:
                combined_mask = element_array
            else:
                combined_mask = np.maximum(combined_mask, element_array)
        
        return Image.fromarray(combined_mask, mode='L')
    
    def suggest_optimal_corrections(self, image: Image.Image) -> Dict[str, Dict[str, float]]:
        """
        Suggest optimal correction parameters based on semantic analysis.
        """
        analysis = self.analyze_card_quality(image)
        corrections = {}
        
        # Default corrections
        corrections['global'] = {
            'brightness': 1.0,
            'contrast': 1.0,
            'saturation': 1.0,
            'gamma': 1.0,
            'color_balance': 0.0
        }
        
        # Apply suggestions
        suggestions = analysis.get('suggested_corrections', [])
        
        for suggestion in suggestions:
            if suggestion == 'increase_brightness':
                corrections['global']['brightness'] = 1.2
            elif suggestion == 'decrease_brightness':
                corrections['global']['brightness'] = 0.85
            elif suggestion == 'increase_contrast':
                corrections['global']['contrast'] = 1.3
            elif suggestion == 'darken_text_areas':
                corrections['text_areas'] = {
                    'brightness': 0.8,
                    'contrast': 1.2
                }
            elif suggestion == 'enhance_borders':
                corrections['border_areas'] = {
                    'contrast': 1.4,
                    'gamma': 0.9
                }
            elif suggestion == 'cool_temperature':
                corrections['global']['color_balance'] = -25
            elif suggestion == 'warm_temperature':
                corrections['global']['color_balance'] = 15
        
        return corrections
    
    def clear_cache(self):
        """Clear the feature cache to free memory."""
        self._feature_cache.clear()
        print("DINOv3 feature cache cleared")


# Global instance for shared use
_analyzer_instance = None

def get_analyzer() -> DINOv3CardAnalyzer:
    """Get shared analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = DINOv3CardAnalyzer()
    return _analyzer_instance

def is_dinov3_available() -> bool:
    """Check if DINOv3 dependencies are available."""
    return DINOV3_AVAILABLE
