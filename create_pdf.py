import os
import re

import click
import inquirer
from utilities import CardSize, PaperSize, generate_pdf

front_directory = os.path.join('game', 'front')
back_directory = os.path.join('game', 'back')
double_sided_directory = os.path.join('game', 'double_sided')
output_directory = os.path.join('game', 'output')

default_output_path = os.path.join(output_directory, 'game.pdf')

def get_available_decks():
    """Get list of available deck folders from the front directory"""
    if not os.path.exists(front_directory):
        return []
    
    decks = []
    for item in os.listdir(front_directory):
        item_path = os.path.join(front_directory, item)
        # only include directories, skip files like .DS_Store and EMPTY.md
        if os.path.isdir(item_path) and not item.startswith('.'):
            decks.append(item)
    
    return sorted(decks)

def select_deck_interactively():
    """Show interactive deck selector"""
    available_decks = get_available_decks()
    
    if not available_decks:
        print("No deck folders found in game/front/")
        return None
    
    questions = [
        inquirer.List(
            'deck',
            message="Select a deck",
            choices=available_decks,
        ),
    ]
    
    answers = inquirer.prompt(questions)
    return answers['deck'] if answers else None

@click.command()
@click.option("--preferred", default=False, is_flag=True, help="Use preferred settings: ppi=800, quality=100, load_offset=True, no_flip_backs=True, extend_corners=15, extend_corners_exclude_borderless=True")
@click.option("--front_dir_path", default=None, help="The path to the directory containing the card fronts. If not specified, uses the default front directory or deck_name subfolder.")
@click.option("--deck_name", help="Name of the deck subfolder within the front directory (e.g. 'budget_tergrid' for game/front/budget_tergrid/). Also used for PDF filename if output_path not specified.")
@click.option("--back_dir_path", default=back_directory, show_default=True, help="The path to the directory containing one or more card backs.")
@click.option("--double_sided_dir_path", default=double_sided_directory, show_default=True, help="The path to the directory containing card backs for double-sided cards.")
@click.option("--output_path", default=None, help="The desired path to the output PDF. If not specified, uses deck_name.pdf or defaults to game.pdf.")
@click.option("--output_images", default=False, is_flag=True, help="Create images instead of a PDF.")
@click.option("--card_size", default=CardSize.STANDARD.value, type=click.Choice([t.value for t in CardSize], case_sensitive=False), show_default=True, help="The desired card size.")
@click.option("--paper_size", default=PaperSize.LETTER.value, type=click.Choice([t.value for t in PaperSize], case_sensitive=False), show_default=True, help="The desired paper size.")
@click.option("--only_fronts", default=False, is_flag=True, help="Only use the card fronts, exclude the card backs.")
@click.option("--crop", help="Crop the outer portion of front and double-sided images. Examples: 3mm, 0.125in, 6.5.")
@click.option("--extend_corners", default=0, type=click.IntRange(min=0), show_default=True, help="Reduce artifacts produced by rounded corners in card images.")
@click.option("--extend_corners_exclude_borderless", default=False, is_flag=True, help="Skip extend_corners processing for borderless cards to preserve their art.")
@click.option("--ppi", default=300, type=click.IntRange(min=0), show_default=True, help="Pixels per inch (PPI) when creating PDF.")
@click.option("--quality", default=75, type=click.IntRange(min=0, max=100), show_default=True, help="File compression. A higher value corresponds to better quality and larger file size.")
@click.option("--load_offset", default=False, is_flag=True, help="Apply saved offsets. See `offset_pdf.py` for more information.")
@click.option("--skip", type=click.IntRange(min=0), multiple=True, help="Skip a card based on its index. Useful for registration issues. Examples: 0, 4.")
@click.option("--name", help="Label each page of the PDF with a name.")
@click.option("--no_flip_backs", default=False, is_flag=True, help="Don't flip card backs 180 degrees. Use for manual printing where you physically flip the paper.")
@click.version_option("1.4.0")

def cli(
    preferred,
    front_dir_path,
    deck_name,
    back_dir_path,
    double_sided_dir_path,
    output_path,
    output_images,
    card_size,
    paper_size,
    only_fronts,
    crop,
    extend_corners,
    extend_corners_exclude_borderless,
    ppi,
    quality,
    skip,
    load_offset,
    name,
    no_flip_backs
):
    # apply preferred settings if --preferred flag is used
    if preferred:
        # override defaults with preferred values
        if ppi == 300:  # only override if user didn't specify custom ppi
            ppi = 800
        if quality == 75:  # only override if user didn't specify custom quality
            quality = 100
        if not load_offset:  # only override if user didn't specify load_offset
            load_offset = True
        if not no_flip_backs:  # only override if user didn't specify no_flip_backs
            no_flip_backs = True
        if extend_corners == 0:  # only override if user didn't specify custom extend_corners
            extend_corners = 15
        if not extend_corners_exclude_borderless:  # only override if user didn't specify this flag
            extend_corners_exclude_borderless = True
        
        # if no deck_name provided with --preferred, show interactive selector
        if not deck_name:
            deck_name = select_deck_interactively()
            if not deck_name:
                print("No deck selected. Exiting.")
                return

    # determine front directory path
    if front_dir_path is None:
        if deck_name:
            front_dir_path = os.path.join(front_directory, deck_name)
        else:
            front_dir_path = front_directory
    
    # if deck_name is provided, also update double_sided directory to use deck-specific folder
    if deck_name and double_sided_dir_path == double_sided_directory:
        double_sided_dir_path = os.path.join(double_sided_directory, deck_name)
    
    # determine output path - use deck name if available and output_path not specified
    if output_path is None:
        if deck_name:
            output_path = os.path.join(output_directory, f'{deck_name}.pdf')
        else:
            output_path = default_output_path
    
    generate_pdf(
        front_dir_path,
        back_dir_path,
        double_sided_dir_path,
        output_path,
        output_images,
        card_size,
        paper_size,
        only_fronts,
        crop,
        extend_corners,
        extend_corners_exclude_borderless,
        ppi,
        quality,
        skip,
        load_offset,
        name,
        no_flip_backs
    )

if __name__ == '__main__':
    cli()