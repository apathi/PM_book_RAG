#!/usr/bin/env python3
"""
Sync processing state with actual data
Rebuilds state files to match reality (chunks_metadata.json)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.state_manager import StateManager, rebuild_state_from_data

def main():
    print("=" * 60)
    print("SYNCING STATE WITH ACTUAL DATA")
    print("=" * 60)

    # Delete old state files
    state_file = Path("data/state/processing_state.json")
    backup_file = Path("data/state/processing_state_backup.json")

    if state_file.exists():
        state_file.unlink()
        print("✅ Deleted old processing_state.json")

    if backup_file.exists():
        backup_file.unlink()
        print("✅ Deleted old processing_state_backup.json")

    # Create fresh state manager (will be empty)
    state_manager = StateManager()
    print("\n📝 Created fresh state manager")

    # Rebuild state from actual data
    print("\n🔄 Rebuilding state from actual data (chunks_metadata.json)...")
    rebuild_state_from_data(state_manager)

    print("\n" + "=" * 60)
    print("✅ STATE SYNC COMPLETE!")
    print("=" * 60)
    print("\n📊 State now matches reality:")
    print("  - Read chunks_metadata.json")
    print("  - Detected processed books")
    print("  - Updated state file")
    print("\n🌐 You can now run: python app.py")
    print("   UI will show correct book status!")

if __name__ == "__main__":
    main()
