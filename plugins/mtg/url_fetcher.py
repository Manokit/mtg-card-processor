import re
import requests
from typing import Optional, Dict, Any
from urllib.parse import urlparse

class DeckURLFetcher:
    """Fetches deck data from MTG websites like Archidekt and Moxfield."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MTG-Card-Processor/1.0'
        })
    
    def extract_deck_id_from_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Extract deck ID and platform from a deck URL.
        
        Returns:
            Dict with 'platform' and 'deck_id' keys, or None if invalid
        """
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path
        
        # Archidekt: https://archidekt.com/decks/14914060/ghostbusters_type_beat
        if 'archidekt.com' in domain:
            match = re.search(r'/decks/(\d+)', path)
            if match:
                return {
                    'platform': 'archidekt',
                    'deck_id': match.group(1)
                }
        
        # Moxfield: https://www.moxfield.com/decks/[deck_id]
        elif 'moxfield.com' in domain:
            match = re.search(r'/decks/([A-Za-z0-9_-]+)', path)
            if match:
                return {
                    'platform': 'moxfield', 
                    'deck_id': match.group(1)
                }
        
        return None
    
    def fetch_archidekt_deck(self, deck_id: str) -> Optional[Dict[str, Any]]:
        """Fetch deck data from Archidekt API."""
        try:
            url = f"https://archidekt.com/api/decks/{deck_id}/"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except Exception as e:
            print(f"Error fetching Archidekt deck {deck_id}: {e}")
            return None
    
    def fetch_moxfield_deck(self, deck_id: str) -> Optional[Dict[str, Any]]:
        """Fetch deck data from Moxfield API."""
        try:
            # Try the public API endpoint used by other tools
            url = f"https://api.moxfield.com/v2/decks/all/{deck_id}"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            return data
            
        except Exception as e:
            print(f"Error fetching Moxfield deck {deck_id}: {e}")
            return None
    
    def convert_archidekt_to_deck_format(self, archidekt_data: Dict[str, Any]) -> str:
        """Convert Archidekt API response to deck text format."""
        if not archidekt_data or 'cards' not in archidekt_data:
            return ""
        
        deck_lines = []
        
        # Process cards
        cards = archidekt_data.get('cards', [])
        
        for card_entry in cards:
            # Extract card information from the Archidekt structure
            quantity = card_entry.get('quantity', 1)
            card_info = card_entry.get('card', {})
            oracle_card = card_info.get('oracleCard', {})
            
            # Get card name from oracle card
            name = oracle_card.get('name', '')
            
            # Get set code from edition
            edition = card_info.get('edition', {})
            set_code = edition.get('editioncode', '').upper()
            
            # Get collector number
            collector_number = card_info.get('collectorNumber', '')
            
            # Get categories for this card
            categories = card_entry.get('categories', [])
            
            if name:
                # Format: 1x Card Name (SET) collector_number [Category]
                line_parts = [f"{quantity}x", name]
                
                if set_code:
                    line_parts.append(f"({set_code})")
                
                if collector_number:
                    line_parts.append(collector_number)
                
                # Add categories as tags if they exist
                if categories:
                    category_str = ",".join(categories)
                    line_parts.append(f"[{category_str}]")
                
                deck_lines.append(" ".join(line_parts))
        
        return "\n".join(deck_lines)
    
    def convert_moxfield_to_deck_format(self, moxfield_data: Dict[str, Any]) -> str:
        """Convert Moxfield API response to deck text format."""
        if not moxfield_data:
            return ""
        
        deck_lines = []
        
        # Process mainboard cards
        mainboard = moxfield_data.get('mainboard', {})
        for card_data in mainboard.values():
            quantity = card_data.get('quantity', 1)
            card = card_data.get('card', {})
            name = card.get('name', '')
            set_code = card.get('set', '')
            collector_number = card.get('cn', '')
            
            if name:
                line_parts = [f"{quantity}", name]
                
                if set_code:
                    line_parts.append(f"({set_code.upper()})")
                
                if collector_number:
                    line_parts.append(collector_number)
                
                deck_lines.append(" ".join(line_parts))
        
        # Process sideboard cards
        sideboard = moxfield_data.get('sideboard', {})
        if sideboard:
            deck_lines.append("\nSIDEBOARD:")
            for card_data in sideboard.values():
                quantity = card_data.get('quantity', 1)
                card = card_data.get('card', {})
                name = card.get('name', '')
                set_code = card.get('set', '')
                collector_number = card.get('cn', '')
                
                if name:
                    line_parts = [f"{quantity}", name]
                    
                    if set_code:
                        line_parts.append(f"({set_code.upper()})")
                    
                    if collector_number:
                        line_parts.append(collector_number)
                    
                    deck_lines.append(" ".join(line_parts))
        
        return "\n".join(deck_lines)
    
    def fetch_deck_from_url(self, url: str) -> Optional[str]:
        """
        Fetch deck data from a URL and return it in deck text format.
        
        Args:
            url: The deck URL (Archidekt or Moxfield)
            
        Returns:
            Deck text in the appropriate format, or None if failed
        """
        deck_info = self.extract_deck_id_from_url(url)
        if not deck_info:
            print(f"Could not parse deck URL: {url}")
            return None
        
        platform = deck_info['platform']
        deck_id = deck_info['deck_id']
        
        print(f"Fetching {platform} deck {deck_id}...")
        
        if platform == 'archidekt':
            data = self.fetch_archidekt_deck(deck_id)
            if data:
                return self.convert_archidekt_to_deck_format(data)
        
        elif platform == 'moxfield':
            data = self.fetch_moxfield_deck(deck_id)
            if data:
                return self.convert_moxfield_to_deck_format(data)
        
        return None
    
    def get_deck_name_from_url(self, url: str) -> Optional[str]:
        """Extract deck name from API response."""
        deck_info = self.extract_deck_id_from_url(url)
        if not deck_info:
            return None
            
        platform = deck_info['platform']
        deck_id = deck_info['deck_id']
        
        if platform == 'archidekt':
            data = self.fetch_archidekt_deck(deck_id)
            if data:
                return data.get('name', f'archidekt_deck_{deck_id}')
        
        elif platform == 'moxfield':
            data = self.fetch_moxfield_deck(deck_id)
            if data:
                return data.get('name', f'moxfield_deck_{deck_id}')
        
        return None