#!/usr/bin/env python3
"""
Process all PM books and create embeddings
Simple, direct approach using atomic BookProcessor and EmbeddingManager
"""
import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.state_manager import StateManager, rebuild_state_from_data
from src.book_processor import BookProcessor
from src.embedding_manager import EmbeddingManager, UnifiedIndexBuilder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Known author mappings (for books where auto-detection fails)
KNOWN_AUTHORS = {
    "cracking-the-pm-career": "Jackie Bavaro, Gayle Laakmann McDowell",
    "the-pm-interview": "Lewis C. Lin",
    "decode-and-conquer": "Lewis C. Lin",
    "the_lean_product_playbook": "Dan Olsen",
    "lean-startup": "Eric Ries"
}


def main():
    """Process all books with atomic state management and recovery"""
    print("=" * 60)
    print("PM Knowledge Assistant - Document Processing Pipeline")
    print("=" * 60)

    # Initialize state manager
    state_manager = StateManager()

    # Sync state with actual data (in case files changed externally)
    print("\n📊 Syncing state with actual data...")
    rebuild_state_from_data(state_manager)

    # Get processing summary
    summary = state_manager.get_processing_summary()
    print(f"\n📚 Books available: {summary['total_books_available']}")
    print(f"✅ Books completed: {summary['books_fully_completed']}")
    print(f"⏳ Books pending: {summary['books_needing_document_processing']}")

    # Check if OpenAI API key is available
    has_api_key = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"
    if not has_api_key:
        print("\n⚠️  No OpenAI API key found. Will process documents only (no embeddings).")
        print("   Set OPENAI_API_KEY in .env file to enable embedding generation.")

    # Process documents
    print("\n" + "=" * 60)
    print("STEP 1: DOCUMENT PROCESSING")
    print("=" * 60)

    book_processor = BookProcessor(state_manager)
    books_to_process = state_manager.get_books_needing_document_processing()

    if not books_to_process:
        print("✅ All books already processed!")
    else:
        print(f"\n📝 Processing {len(books_to_process)} books...")

        processed_count = 0
        failed_count = 0

        for book_name in books_to_process:
            try:
                print(f"\n📚 Processing: {book_name}")
                result = book_processor.process_book_atomic(book_name)

                if not result.errors:
                    print(f"   ✅ Success: {result.total_chunks} chunks from {result.total_chapters} chapters")
                    processed_count += 1
                else:
                    print(f"   ❌ Failed: {', '.join(result.errors)}")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error processing {book_name}: {e}")
                print(f"   ❌ Error: {str(e)}")
                failed_count += 1
                continue

        print(f"\n📊 Document Processing Summary:")
        print(f"   Processed: {processed_count}")
        print(f"   Failed: {failed_count}")

    # Process embeddings
    if has_api_key:
        print("\n" + "=" * 60)
        print("STEP 2: EMBEDDING GENERATION")
        print("=" * 60)

        embedding_manager = EmbeddingManager(state_manager)
        books_needing_embeddings = state_manager.get_books_needing_embedding_processing()

        if not books_needing_embeddings:
            print("✅ All books already have embeddings!")
        else:
            print(f"\n🔮 Generating embeddings for {len(books_needing_embeddings)} books...")

            embedding_success = 0
            embedding_failed = 0

            for book_name in books_needing_embeddings:
                try:
                    print(f"\n🔮 Embedding: {book_name}")
                    result = embedding_manager.process_book_embeddings(book_name)

                    if result.get('success'):
                        chunks_count = result.get('successful_count', 0)
                        print(f"   ✅ Generated embeddings for {chunks_count} chunks")
                        embedding_success += 1
                    else:
                        error = result.get('error', 'Unknown error')
                        print(f"   ❌ Failed: {error}")
                        embedding_failed += 1

                except Exception as e:
                    logger.error(f"Error generating embeddings for {book_name}: {e}")
                    print(f"   ❌ Error: {str(e)}")
                    embedding_failed += 1
                    continue

            print(f"\n📊 Embedding Generation Summary:")
            print(f"   Success: {embedding_success}")
            print(f"   Failed: {embedding_failed}")

    # Build unified index
    print("\n" + "=" * 60)
    print("STEP 3: BUILD UNIFIED INDEX")
    print("=" * 60)

    try:
        print("\n🔨 Building unified index from all completed books...")
        index_builder = UnifiedIndexBuilder(state_manager)
        result = index_builder.build_unified_index()

        if result.get('success'):
            total_chunks = result.get('total_chunks', 0)
            books_included = result.get('books_included', [])
            print(f"✅ Unified index built successfully!")
            print(f"   Total chunks: {total_chunks}")
            print(f"   Books included: {len(books_included)}")
        else:
            print(f"❌ Failed to build unified index: {result.get('error')}")

    except Exception as e:
        logger.error(f"Error building unified index: {e}")
        print(f"❌ Error: {str(e)}")

    # Final summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)

    final_summary = state_manager.get_processing_summary()
    print(f"\n📊 Final Status:")
    print(f"   Books fully completed: {final_summary['books_fully_completed']}/{final_summary['total_books_available']}")
    print(f"   Total chunks in index: {final_summary.get('unified_index_total_chunks', 0)}")

    # Validate system consistency
    validation = state_manager.validate_consistency()
    if validation['consistent']:
        print(f"\n✅ System validation: PASSED")
    else:
        print(f"\n⚠️  System validation found {len(validation['issues'])} issues:")
        for issue in validation['issues'][:5]:
            print(f"   - {issue}")

    print("\n🎯 Next Steps:")
    print("   1. Run 'python app.py' to start the web interface")
    print("   2. Use 'python utils/sync_state.py' to rebuild state if needed")
    print("   3. Check 'data/state/processing_state.json' for detailed status")

    print("\n💡 Recovery Features:")
    print("   ✅ Atomic per-book processing - safe to interrupt anytime")
    print("   ✅ Automatic resume - rerun this script to continue")
    print("   ✅ State validation - consistent data guaranteed")


if __name__ == "__main__":
    main()
