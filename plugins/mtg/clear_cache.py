#!/usr/bin/env python3
"""
utility script to manage mtg card image cache
"""

import click
from card_selector_gui import CardSelectorGUI

@click.command()
@click.option('--stats', is_flag=True, help='Show cache statistics')
@click.option('--clear', is_flag=True, help='Clear all cached images')
def main(stats, clear):
    """manage mtg card image cache"""
    
    if clear:
        CardSelectorGUI.clear_all_cache()
        return
        
    if stats or (not clear):
        # show stats by default
        cache_stats = CardSelectorGUI.get_cache_stats()
        print(f"mtg card image cache statistics:")
        print(f"  memory cache: {cache_stats['memory_images']} images ({cache_stats['memory_size_mb']:.1f} mb)")
        print(f"  disk cache: {cache_stats['disk_images']} images ({cache_stats['disk_size_mb']:.1f} mb)")
        print(f"  cache directory: {cache_stats['cache_dir']}")
        
        if cache_stats['memory_images'] > 0 or cache_stats['disk_images'] > 0:
            print(f"  total cache benefit: {cache_stats['memory_images'] + cache_stats['disk_images']} images")
            print(f"  run with --clear to reset cache")

if __name__ == '__main__':
    main() 