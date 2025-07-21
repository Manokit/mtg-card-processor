import os
from typing import List, Set, Tuple, Optional
import re
import requests
import time
from urllib.parse import quote
from card_selector_gui import CardSelectorGUI

double_sided_layouts = ['transform', 'modal_dfc']

def request_scryfall(
    query: str,
) -> requests.Response:
    r = requests.get(query, headers = {'user-agent': 'silhouette-card-maker/0.1', 'accept': '*/*'})

    # Check for 2XX response code
    r.raise_for_status()

    # Sleep for 150 milliseconds, greater than the 100ms requested by Scryfall API documentation
    time.sleep(0.15)

    return r

def fetch_card_art(
    index: int,
    quantity: int,

    clean_card_name: str,
    card_set: int,
    card_collector_number: int,
    layout: str,

    front_img_dir: str,
    double_sided_dir: str
) -> None:
    # Query for the front side
    card_front_image_query = f'https://api.scryfall.com/cards/{card_set}/{card_collector_number}/?format=image&version=png'
    card_art = request_scryfall(card_front_image_query).content
    if card_art is not None:

        # Save image based on quantity
        for counter in range(quantity):
            image_path = os.path.join(front_img_dir, f'{str(index)}{clean_card_name}{str(counter + 1)}.png')

            with open(image_path, 'wb') as f:
                f.write(card_art)

    # Get backside of card, if it exists
    if layout in double_sided_layouts:
        card_back_image_query = f'{card_front_image_query}&face=back'
        card_art = request_scryfall(card_back_image_query).content
        if card_art is not None:

            # Save image based on quantity
            for counter in range(quantity):
                image_path = os.path.join(double_sided_dir, f'{str(index)}{clean_card_name}{str(counter + 1)}.png')

                with open(image_path, 'wb') as f:
                    f.write(card_art)

def remove_nonalphanumeric(s: str) -> str:
    return re.sub(r'[^\w]', '', s)

def extract_front_face_name(card_name: str) -> str:
    """extract front face name from double-sided cards (split on //)"""
    if '//' in card_name:
        return card_name.split('//')[0].strip()
    return card_name

def partition_printings(printings: List, condition: List) -> Tuple[List, List]:
    matches = []
    non_matches = []
    for card in printings:
        (matches if condition(card) else non_matches).append(card)
    return matches, non_matches

def progressive_filtering(printings: List, filters):
    pool = printings
    leftovers = []

    for condition in filters:
        matched, not_matched = partition_printings(pool, condition)
        leftovers = not_matched + leftovers
        pool = matched or pool  # Only narrow if we have any matches

    return pool + leftovers

def filtering(printings: List, filters):
    pool = printings

    for condition in filters:
        matched, _ = partition_printings(pool, condition)
        pool = matched

    return pool

def fetch_card(
    index: int,
    quantity: int,

    card_set: str,
    card_collector_number: str,
    ignore_set_and_collector_number: bool,

    name: str,

    prefer_older_sets: bool,
    preferred_sets: Set[str],

    prefer_showcase: bool,
    prefer_extra_art: bool,

    front_img_dir: str,
    double_sided_dir: str
):
    if not ignore_set_and_collector_number and card_set != "" and card_collector_number != "":
        card_info_query = f"https://api.scryfall.com/cards/{card_set}/{card_collector_number}"

        # Query for card info
        card_json = request_scryfall(card_info_query).json()

        fetch_card_art(index, quantity, remove_nonalphanumeric(card_json['name']), card_set, card_collector_number, card_json['layout'], front_img_dir, double_sided_dir)

    else:
        if name == "":
            raise Exception()

        # Filter out symbols from card names - use front face name for double-sided cards  
        front_face_name = extract_front_face_name(name)

        # debug output for double-sided cards
        if '//' in name:
            print(f'double-sided card detected: "{name}" -> searching for: "{front_face_name}"')

        # url encode the name for api search
        encoded_name = quote(front_face_name)
        card_info_query = f'https://api.scryfall.com/cards/named?exact={encoded_name}'

        # Query for card info
        try:
            card_json = request_scryfall(card_info_query).json()
        except Exception as e:
            # if exact search fails and this is a double-sided card, try fuzzy search
            if '//' in name:
                print(f'exact search failed for double-sided card, trying fuzzy search...')
                fuzzy_query = f'https://api.scryfall.com/cards/named?fuzzy={encoded_name}'
                try:
                    print(f'fuzzy searching scryfall for: {fuzzy_query}')
                    card_json = request_scryfall(fuzzy_query).json()
                except Exception as e2:
                    print(f'fuzzy search also failed for "{front_face_name}": {e2}')
                    raise e2
            else:
                print(f'failed to find card "{front_face_name}": {e}')
                raise e

        set = card_json["set"]
        collector_number = card_json["collector_number"]

        # If preferred options are used, then filter over prints
        if prefer_older_sets or len(preferred_sets) > 0 or prefer_showcase or prefer_extra_art:
            # Get available printings
            prints_search_json = request_scryfall(card_json['prints_search_uri']).json()
            card_printings = prints_search_json['data']

            # Optional reverse for older preferences
            if prefer_older_sets:
                card_printings.reverse()

            # Define filters in order of preference
            filters = [
                lambda c: c['nonfoil'],
                lambda c: not c['digital'],
                lambda c: not c['promo'],
                lambda c: c['set'] in preferred_sets,
                lambda c: not prefer_showcase ^ ('frame_effects' in c and 'showcase' in c['frame_effects']),
                lambda c: not prefer_extra_art ^ (c['full_art'] or c['border_color'] == "borderless" or ('frame_effects' in c and 'extendedart' in c['frame_effects']))
            ]

            # Apply progressive filtering
            filtered_printings = progressive_filtering(card_printings, filters)

            if len(filtered_printings) == 0:
                print(f'No printings found for "{name}" with preferred options. Using default instead.')
            else:
                best_print = filtered_printings[0]
                set = best_print["set"]
                collector_number = best_print["collector_number"]

        # Fetch card art - use cleaned name for file naming
        clear_card_name = remove_nonalphanumeric(card_json['name'])
        fetch_card_art(
            index,
            quantity,
            clear_card_name,
            set,
            collector_number,
            card_json['layout'],
            front_img_dir,
            double_sided_dir
        )

def fetch_card_with_gui(
    index: int,
    quantity: int,

    card_set: str,
    card_collector_number: str,
    ignore_set_and_collector_number: bool,

    name: str,

    front_img_dir: str,
    double_sided_dir: str,
    gui: CardSelectorGUI
) -> bool:
    """
    fetch card with gui selection, returns true if card was selected and downloaded
    """
    if not ignore_set_and_collector_number and card_set != "" and card_collector_number != "":
        # if we have specific set/collector number, just fetch that directly
        card_info_query = f"https://api.scryfall.com/cards/{card_set}/{card_collector_number}"
        card_json = request_scryfall(card_info_query).json()
        
        fetch_card_art(
            index, quantity, remove_nonalphanumeric(card_json['name']), 
            card_set, card_collector_number, card_json['layout'], 
            front_img_dir, double_sided_dir
        )
        return True
    else:
        if name == "":
            return False

        # get card info and all printings - use front face name for double-sided cards
        front_face_name = extract_front_face_name(name)
        
        # debug output for double-sided cards
        if '//' in name:
            print(f'double-sided card detected: "{name}" -> searching for: "{front_face_name}"')
        
        # try exact search first, then fuzzy search for double-sided cards
        encoded_name = quote(front_face_name)
        card_info_query = f'https://api.scryfall.com/cards/named?exact={encoded_name}'
        
        try:
            print(f'searching scryfall for: {card_info_query}')
            card_json = request_scryfall(card_info_query).json()
        except Exception as e:
            # if exact search fails and this is a double-sided card, try fuzzy search
            if '//' in name:
                print(f'exact search failed for double-sided card, trying fuzzy search...')
                fuzzy_query = f'https://api.scryfall.com/cards/named?fuzzy={encoded_name}'
                try:
                    print(f'fuzzy searching scryfall for: {fuzzy_query}')
                    card_json = request_scryfall(fuzzy_query).json()
                except Exception as e2:
                    print(f'fuzzy search also failed for "{front_face_name}": {e2}')
                    print(f'original card name was: "{name}"')
                    return False
            else:
                print(f'failed to find card "{front_face_name}": {e}')
                print(f'original card name was: "{name}"')
                return False

        # get all available printings
        prints_search_json = request_scryfall(card_json['prints_search_uri']).json()
        card_printings = prints_search_json['data']

        # filter out digital-only and invalid printings
        valid_printings = []
        for printing in card_printings:
            # skip if no image available
            if 'image_uris' not in printing or 'normal' not in printing['image_uris']:
                continue
            valid_printings.append(printing)

        if not valid_printings:
            print(f'no valid printings found for "{name}"')
            return False

        # show gui selection dialog (use original name for display)
        selected_printing = gui.show_selection_dialog(name, valid_printings)
        
        if selected_printing is None:
            print(f'skipped card "{name}"')
            return False

        # fetch the selected card art
        # use the actual card name from scryfall for file naming
        actual_card_name = remove_nonalphanumeric(selected_printing['name'])
        fetch_card_art(
            index,
            quantity,
            actual_card_name,
            selected_printing['set'],
            selected_printing['collector_number'],
            selected_printing['layout'],
            front_img_dir,
            double_sided_dir
        )
        return True

def get_handle_card(
    ignore_set_and_collector_number: bool,

    prefer_older_sets: bool,
    preferred_sets: Set[str],

    prefer_showcase: bool,
    prefer_extra_art: bool,

    front_img_dir: str,
    double_sided_dir: str,
    use_gui: bool = False
):
    if use_gui:
        gui = CardSelectorGUI()
        
        def configured_fetch_card_gui(index: int, name: str, card_set: str = None, card_collector_number: int = None, quantity: int = 1):
            return fetch_card_with_gui(
                index,
                quantity,

                card_set,
                card_collector_number,
                ignore_set_and_collector_number,

                name,

                front_img_dir,
                double_sided_dir,
                gui
            )
        return configured_fetch_card_gui
    else:
        def configured_fetch_card(index: int, name: str, card_set: str = None, card_collector_number: int = None, quantity: int = 1):
            fetch_card(
                index,
                quantity,

                card_set,
                card_collector_number,
                ignore_set_and_collector_number,

                name,

                prefer_older_sets,
                preferred_sets,

                prefer_showcase,
                prefer_extra_art,

                front_img_dir,
                double_sided_dir
            )
        return configured_fetch_card