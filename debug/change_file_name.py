import os
import shutil
import re
from argparse import ArgumentParser
from collections import defaultdict


def parse_folder_name(folder_name):
    """
    Parse folder name to extract prefix and number suffix.
    
    Args:
        folder_name: Name of the folder
        
    Returns:
        tuple: (prefix, number) or (None, None) if no number found
    """
    # Find the last sequence of digits at the end of the folder name
    match = re.search(r'^(.+?)(\d+)$', folder_name)
    if match:
        prefix = match.group(1)
        number = int(match.group(2))
        return prefix, number
    return None, None


def find_folders_with_prefix(data_root, old_prefix):
    """
    Find all folders that start with the given prefix.
    
    Args:
        data_root: Root directory to search
        old_prefix: Prefix to search for
        
    Returns:
        list: List of (folder_path, folder_name, prefix, number) tuples
    """
    matching_folders = []
    
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        
        if not os.path.isdir(folder_path):
            continue
            
        # Check if folder starts with the old prefix
        if folder_name.startswith(old_prefix):
            # Parse to get the number suffix
            parsed_prefix, number = parse_folder_name(folder_name)
            if parsed_prefix == old_prefix and number is not None:
                matching_folders.append((folder_path, folder_name, parsed_prefix, number))
    
    return matching_folders


def rename_folders(data_root, old_prefix, new_prefix, dry_run=True):
    """
    Rename folders from old_prefix to new_prefix while preserving number suffixes.
    
    Args:
        data_root: Root directory containing folders
        old_prefix: Current prefix to replace
        new_prefix: New prefix to use
        dry_run: If True, only show what would be changed without actually changing
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Find all matching folders
    matching_folders = find_folders_with_prefix(data_root, old_prefix)
    
    if not matching_folders:
        print(f"No folders found with prefix '{old_prefix}'")
        return False
    
    # Sort by number for consistent ordering
    matching_folders.sort(key=lambda x: x[3])
    
    print(f"Found {len(matching_folders)} folders with prefix '{old_prefix}':")
    for _, folder_name, _, number in matching_folders:
        print(f"  {folder_name}")
    
    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN MODE - No changes will be made")
        print("Use --execute to actually rename folders")
    else:
        print("EXECUTING - Folders will be renamed")
    print(f"{'='*60}")
    
    # Check for conflicts
    conflicts = []
    for _, folder_name, _, number in matching_folders:
        new_name = f"{new_prefix}{number}"
        new_path = os.path.join(data_root, new_name)
        
        if os.path.exists(new_path):
            conflicts.append((folder_name, new_name))
    
    if conflicts:
        print(f"\n⚠️  WARNING: {len(conflicts)} potential conflicts detected:")
        for old_name, new_name in conflicts:
            print(f"  '{old_name}' -> '{new_name}' (target already exists)")
        
        if not dry_run:
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Operation cancelled.")
                return False
    
    # Perform renaming
    success_count = 0
    error_count = 0
    
    for folder_path, folder_name, _, number in matching_folders:
        new_name = f"{new_prefix}{number}"
        new_path = os.path.join(data_root, new_name)
        
        print(f"\nRenaming: '{folder_name}' -> '{new_name}'")
        
        if dry_run:
            print(f"  Would move: {folder_path}")
            print(f"  To:         {new_path}")
            success_count += 1
        else:
            try:
                # Use shutil.move for cross-platform compatibility
                shutil.move(folder_path, new_path)
                print(f"  ✅ Successfully renamed")
                success_count += 1
            except Exception as e:
                print(f"  ❌ Error: {e}")
                error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN SUMMARY:")
        print(f"  Would rename: {success_count} folders")
    else:
        print("RENAME SUMMARY:")
        print(f"  Successfully renamed: {success_count} folders")
        if error_count > 0:
            print(f"  Errors: {error_count} folders")
    print(f"{'='*60}")
    
    return error_count == 0


def preview_changes(data_root, old_prefix, new_prefix):
    """
    Show a preview of what would be changed without making changes.
    
    Args:
        data_root: Root directory containing folders
        old_prefix: Current prefix to replace
        new_prefix: New prefix to use
    """
    matching_folders = find_folders_with_prefix(data_root, old_prefix)
    
    if not matching_folders:
        print(f"No folders found with prefix '{old_prefix}'")
        return
    
    # Sort by number
    matching_folders.sort(key=lambda x: x[3])
    
    print(f"\n{'='*80}")
    print(f"PREVIEW: Renaming folders from '{old_prefix}' to '{new_prefix}'")
    print(f"{'='*80}")
    print(f"Found {len(matching_folders)} folders to rename:")
    print()
    
    # Check for conflicts
    conflicts = []
    for _, folder_name, _, number in matching_folders:
        new_name = f"{new_prefix}{number}"
        new_path = os.path.join(data_root, new_name)
        
        if os.path.exists(new_path):
            conflicts.append((folder_name, new_name))
    
    # Show renaming plan
    for _, folder_name, _, number in matching_folders:
        new_name = f"{new_prefix}{number}"
        status = "⚠️  CONFLICT" if any(c[1] == new_name for c in conflicts) else "✅ OK"
        print(f"{status} {folder_name:30s} -> {new_name}")
    
    if conflicts:
        print(f"\n⚠️  {len(conflicts)} conflicts detected - some target names already exist")
        print("   You may need to handle these manually or choose a different new prefix")
    
    print(f"\n{'='*80}")


def main():
    parser = ArgumentParser(description="Rename folders while preserving number suffixes")
    parser.add_argument("--data_root", type=str, 
                       default="/home/robot/drive/robotool/videos_0831",
                       help="Root directory containing folders to rename")
    parser.add_argument("--old_prefix", type=str, default='475f0955_spoon_cut_tofu_50_',
                       help="Current prefix to replace")
    parser.add_argument("--new_prefix", type=str, default='spoon_cut_tofu_50_big_',
                       help="New prefix to use")
    parser.add_argument("--preview", action="store_true",
                       help="Show preview of changes without making them")
    parser.add_argument("--execute", action="store_true",
                       help="Actually execute the renaming (default is dry-run)")
    parser.add_argument("--force", action="store_true",
                       help="Force execution even if conflicts are detected")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory '{args.data_root}' does not exist.")
        return
    
    # Validate prefixes
    if not args.old_prefix or not args.new_prefix:
        print("Error: Both old_prefix and new_prefix are required.")
        return
    
    if args.old_prefix == args.new_prefix:
        print("Error: Old and new prefixes are the same. No changes needed.")
        return
    
    print(f"Directory: {args.data_root}")
    print(f"Old prefix: '{args.old_prefix}'")
    print(f"New prefix: '{args.new_prefix}'")
    
    if args.preview:
        # Preview mode
        preview_changes(args.data_root, args.old_prefix, args.new_prefix)
    else:
        # Rename mode
        dry_run = not args.execute
        success = rename_folders(args.data_root, args.old_prefix, args.new_prefix, dry_run)
        
        if success and dry_run:
            print(f"\nTo execute these changes, run with --execute flag:")
            print(f"python change_file_name.py --data_root {args.data_root} --old_prefix '{args.old_prefix}' --new_prefix '{args.new_prefix}' --execute")


if __name__ == "__main__":
    main()
