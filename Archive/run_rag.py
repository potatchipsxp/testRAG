#!/usr/bin/env python3
"""
Main script to run the RAG system.

Usage:
    python run_rag.py              # Run demo queries
    python run_rag.py --index      # Re-index documents
    python run_rag.py --interactive  # Interactive query mode
"""

import sys
from rag_functions import (
    index_documents,
    query,
    get_collection_stats,
    reset_collection
)


def run_demo_queries():
    """Run a set of demo queries to showcase the system."""
    
    # Check collection status
    stats = get_collection_stats()
    print(f"Collection: {stats['collection_name']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Sample sources: {stats['sample_sources'][:5]}")
    print()
    
    if stats['total_chunks'] == 0:
        print("Collection is empty! Indexing documents...")
        index_documents("./data")
        print()
    
    # Demo questions
    questions = [
        "When was Abraham Lincoln born?",
        "What were Theodore Roosevelt's major accomplishments?",
        "Which presidents served during World War II?",
    ]
    
    for i, question in enumerate(questions, 1):
        result = query(question, n_results=5, verbose=True)
        
        print(f"\nANSWER:")
        print("-" * 70)
        print(result['answer'])
        print("-" * 70)
        
        if i < len(questions):
            input("\nPress Enter for next question...")
            print("\n")


def run_interactive():
    """Interactive query mode - ask questions in a loop."""
    
    # Check collection status
    stats = get_collection_stats()
    print(f"Collection: {stats['collection_name']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print()
    
    if stats['total_chunks'] == 0:
        print("Collection is empty! Indexing documents...")
        index_documents("./data")
        print()
    
    print("=" * 70)
    print("INTERACTIVE RAG QUERY MODE")
    print("=" * 70)
    print("Ask questions about US Presidents.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'stats' to see collection statistics.")
    print("=" * 70)
    print()
    
    while True:
        try:
            # Get user input
            question = input("\nYour question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'stats':
                stats = get_collection_stats()
                print(f"\nCollection Statistics:")
                print(f"  Total chunks: {stats['total_chunks']}")
                print(f"  Sources: {', '.join(stats['sample_sources'])}")
                continue
            
            # Query the RAG system
            result = query(question, n_results=5, verbose=True)
            
            print(f"\nANSWER:")
            print("-" * 70)
            print(result['answer'])
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def do_indexing():
    """Index or re-index documents."""
    
    print("This will index all documents in ./data/")
    response = input("Proceed? (y/n): ").strip().lower()
    
    if response == 'y':
        stats = get_collection_stats()
        if stats['total_chunks'] > 0:
            print(f"\nCollection already has {stats['total_chunks']} chunks.")
            response = input("Delete and re-index? (y/n): ").strip().lower()
            if response == 'y':
                reset_collection()
        
        index_documents("./data")
        print("\nIndexing complete!")
        
        # Show stats
        stats = get_collection_stats()
        print(f"\nCollection Statistics:")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Sources: {', '.join(stats['sample_sources'])}")
    else:
        print("Cancelled.")


def main():
    """Main entry point."""
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['--index', '-i']:
            do_indexing()
        elif command in ['--interactive', '-int']:
            run_interactive()
        elif command in ['--help', '-h']:
            print(__doc__)
        else:
            print(f"Unknown command: {command}")
            print("Use --help for usage information")
    else:
        # Default: run demo queries
        run_demo_queries()


if __name__ == "__main__":
    main()
