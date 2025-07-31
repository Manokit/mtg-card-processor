import os

import click
from deck_formats import DeckFormat, parse_deck
from scryfall import get_handle_card
from url_fetcher import DeckURLFetcher

from typing import Set

front_directory = os.path.join('game', 'front')
double_sided_directory = os.path.join('game', 'double_sided')

@click.command()
@click.argument('deck_source')
@click.argument('format', type=click.Choice([t.value for t in DeckFormat], case_sensitive=False), required=False)
@click.option('-i', '--ignore_set_and_collector_number', default=False, is_flag=True, show_default=True, help="Ignore provided sets and collector numbers when fetching cards.")
@click.option('--prefer_older_sets', default=False, is_flag=True, show_default=True, help="Prefer fetching cards from older sets if sets are not provided.")
@click.option('-s', '--prefer_set', multiple=True, help="Prefer fetching cards from a particular set(s) if sets are not provided. Use this option multiple times to specify multiple preferred sets.")
@click.option('--prefer_showcase', default=False, is_flag=True, show_default=True, help="Prefer fetching cards with showcase treatment")
@click.option('--prefer_extra_art', default=False, is_flag=True, show_default=True, help="Prefer fetching cards with full art, borderless, or extended art.")
@click.option('--gui', default=False, is_flag=True, show_default=True, help="Show GUI for manual card art selection with navigation controls")

def cli(
    deck_source: str,
    format: DeckFormat,
    ignore_set_and_collector_number: bool,
    prefer_older_sets: bool,
    prefer_set: Set[str],

    prefer_showcase: bool,
    prefer_extra_art: bool,
    gui: bool
):
    # Check if deck_source is a URL or file path
    is_url = deck_source.startswith(('http://', 'https://')) or any(domain in deck_source for domain in ['archidekt.com', 'moxfield.com'])
    
    if is_url:
        # Handle URL input
        url_fetcher = DeckURLFetcher()
        
        # Extract deck name for directory structure
        deck_name = url_fetcher.get_deck_name_from_url(deck_source)
        if not deck_name:
            print(f'Could not extract deck name from URL: {deck_source}')
            return
        
        # Fetch deck data from URL
        deck_text = url_fetcher.fetch_deck_from_url(deck_source)
        if not deck_text:
            print(f'Failed to fetch deck data from URL: {deck_source}')
            return
        
        # Determine format automatically based on URL
        deck_info = url_fetcher.extract_deck_id_from_url(deck_source)
        if not deck_info:
            print(f'Could not determine platform from URL: {deck_source}')
            return
        
        platform = deck_info['platform']
        if platform == 'archidekt':
            detected_format = DeckFormat.ARCHIDEKT
        elif platform == 'moxfield':  
            detected_format = DeckFormat.MOXFIELD
        else:
            detected_format = format or DeckFormat.SIMPLE
        
        # Use detected format if none provided
        if format is None:
            format = detected_format
            print(f'Auto-detected format: {format.value}')
        
        print(f'Successfully fetched deck "{deck_name}" from {platform}')
        
    else:
        # Handle file path input
        if not os.path.isfile(deck_source):
            print(f'{deck_source} is not a valid file.')
            return

        # extract deck name from file path and create subfolder
        deck_filename = os.path.basename(deck_source)
        deck_name = os.path.splitext(deck_filename)[0]  # remove file extension
        
        # Require format for file input
        if format is None:
            print('Format is required when using a file path. Please specify the deck format.')
            return
        
        with open(deck_source, 'r') as deck_file:
            deck_text = deck_file.read()
    
    # create deck-specific directories
    deck_front_directory = os.path.join(front_directory, deck_name)
    deck_double_sided_directory = os.path.join(double_sided_directory, deck_name)
    
    # ensure directories exist
    os.makedirs(deck_front_directory, exist_ok=True)
    os.makedirs(deck_double_sided_directory, exist_ok=True)
    
    print(f'Fetching cards for deck "{deck_name}" into: {deck_front_directory}')

    parse_deck(
        deck_text,
        format,
        get_handle_card(
            ignore_set_and_collector_number,

            prefer_older_sets,
            prefer_set,
            
            prefer_showcase,
            prefer_extra_art,

            deck_front_directory,
            deck_double_sided_directory,
            gui
        )
    )

if __name__ == '__main__':
    cli()