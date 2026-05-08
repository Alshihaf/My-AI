"""
File Manager - Handles all file system operations for Samre.

This module provides a concrete implementation for reading, writing, and listing
files, replacing the abstract default_api with real OS interactions.
"""

import os
from pathlib import Path
from typing import List, Dict, Union

class FileManager:
    """
    Manages file operations such as reading, writing, listing, and deleting files.
    Provides a structured interface to the filesystem for other components.
    """
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path).resolve()
        print(f"📦 FileManager initialized. Base path: {self.base_path}")

    def _to_absolute_path(self, relative_path: Union[str, Path]) -> Path:
        """Converts a relative path to a secure, absolute path within the base directory."""
        # This prevents directory traversal attacks (e.g., using "../")
        return self.base_path.joinpath(relative_path).resolve()

    def list_files(self, path: str = ".") -> Dict[str, Union[str, List[str]]]:
        """
        Lists all files and directories in a given path relative to the base path.
        
        Args:
            path: The relative path of the directory to inspect.

        Returns:
            A dictionary containing lists of files and directories.
        """
        try:
            target_path = self._to_absolute_path(path)
            if not target_path.is_dir():
                return {"error": f"Path '{path}' is not a valid directory."}

            files = [f.name for f in target_path.iterdir() if f.is_file()]
            directories = [d.name for d in target_path.iterdir() if d.is_dir()]
            
            return {"files": files, "directories": directories}
        except Exception as e:
            return {"error": f"Failed to list files in '{path}': {e}"}

    def read_file(self, path: str) -> Dict[str, str]:
        """
        Reads the content of a file.

        Args:
            path: The relative path of the file to read.

        Returns:
            A dictionary with the file content or an error.
        """
        try:
            file_path = self._to_absolute_path(path)
            if not file_path.is_file():
                return {"error": f"File not found: {path}"}

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {"content": content}
        except Exception as e:
            return {"error": f"Failed to read file '{path}': {e}"}

    def write_file(self, path: str, content: str) -> Dict[str, str]:
        """
        Creates or overwrites a file with new content.

        Args:
            path: The relative path of the file to write.
            content: The content to write to the file.

        Returns:
            A dictionary indicating success or failure.
        """
        try:
            file_path = self._to_absolute_path(path)
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {"status": f"File '{path}' written successfully."}
        except Exception as e:
            return {"error": f"Failed to write to file '{path}': {e}"}

    def delete_file(self, path: str) -> Dict[str, str]:
        """
        Deletes a file.

        Args:
            path: The relative path of the file to delete.

        Returns:
            A dictionary indicating success or failure.
        """
        try:
            file_path = self._to_absolute_path(path)
            if not file_path.is_file():
                return {"error": f"File not found: {path}"}
            
            file_path.unlink()
            return {"status": f"File '{path}' deleted successfully."}
        except Exception as e:
            return {"error": f"Failed to delete file '{path}': {e}"}
