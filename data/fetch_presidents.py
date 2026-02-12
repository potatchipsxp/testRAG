#!/usr/bin/env python3
"""
Fetch Wikipedia articles about US Presidents and save them as text files.
"""

import wikipediaapi
import os

def fetch_president_articles():
    """Download Wikipedia articles for selected US Presidents."""
    
    # Initialize Wikipedia API with a user agent
    wiki = wikipediaapi.Wikipedia(
        user_agent='RAGTestProject/1.0',
        language='en'
    )
    
    # List of presidents to fetch (mix of different eras)
    presidents = [
        "George Washington",
        "Thomas Jefferson",
        "Abraham Lincoln",
        "Theodore Roosevelt",
        "Franklin D. Roosevelt",
        "John F. Kennedy",
        "Ronald Reagan",
        "Barack Obama",
        "Donald Trump",
        "Joe Biden"
    ]
    
    print("Fetching Wikipedia articles for US Presidents...")
    print("-" * 50)
    
    for president in presidents:
        try:
            # Fetch the page
            page = wiki.page(president)
            
            if page.exists():
                # Create a safe filename
                filename = president.replace(" ", "_").replace(".", "") + ".txt"
                filepath = os.path.join(".", filename)
                
                # Write the content
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {page.title}\n")
                    f.write(f"URL: {page.fullurl}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(page.text)
                
                print(f"✓ Downloaded: {president} ({len(page.text):,} characters)")
            else:
                print(f"✗ Not found: {president}")
                
        except Exception as e:
            print(f"✗ Error fetching {president}: {e}")
    
    print("-" * 50)
    print("Done! Articles saved to current directory.")

if __name__ == "__main__":
    fetch_president_articles()
