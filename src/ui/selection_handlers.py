"""
Selection Handlers for UI - Clean UI interaction logic
Handles all user interactions with book/chapter selection
"""
import gradio as gr
import logging
from typing import List, Tuple, Callable, Any

from .source_manager import SourceManager, BookInfo

logger = logging.getLogger(__name__)


class SelectionHandlers:
    """
    Clean handlers for UI selection interactions
    
    Features:
    - Smart selection management
    - Real-time feedback
    - Error handling
    - Clean separation from UI components
    """
    
    def __init__(self, source_manager: SourceManager):
        self.source_manager = source_manager
        self._cached_books: List[BookInfo] = []
    
    def refresh_sources(self) -> Tuple[gr.CheckboxGroup, str]:
        """
        Refresh available sources and return updated UI components
        
        Returns:
            Tuple of (updated_checkbox_group, status_display)
        """
        try:
            # Get fresh book data
            self._cached_books = self.source_manager.get_available_books()
            
            # Format for checkbox choices
            choices = self.source_manager.format_book_selection_choices(self._cached_books)
            
            # Get smart defaults
            defaults = self.source_manager.get_default_selections(self._cached_books)
            
            # Create status display
            status_display = self.source_manager.format_status_display(self._cached_books)
            
            logger.info(f"Refreshed sources: {len(choices)} books available, {len(defaults)} auto-selected")
            
            return (
                gr.CheckboxGroup(choices=choices, value=defaults),
                status_display
            )
            
        except Exception as e:
            logger.error(f"Error refreshing sources: {e}")
            return (
                gr.CheckboxGroup(choices=[], value=[]),
                f"❌ Error loading sources: {str(e)}"
            )
    
    def select_all_available(self) -> List[str]:
        """Select all available books"""
        try:
            if not self._cached_books:
                self._cached_books = self.source_manager.get_available_books()
            
            available_books = self.source_manager.filter_available_books(self._cached_books)
            all_names = [book.name for book in available_books]
            
            logger.info(f"Selected all {len(all_names)} available books")
            return all_names
            
        except Exception as e:
            logger.error(f"Error selecting all sources: {e}")
            return []
    
    def clear_all_selections(self) -> List[str]:
        """Clear all selections"""
        logger.info("Cleared all selections")
        return []
    
    def update_selection_summary(self, selected_books: List[str]) -> str:
        """
        Update selection summary display
        
        Args:
            selected_books: List of selected book names
            
        Returns:
            Formatted selection summary
        """
        try:
            if not self._cached_books:
                self._cached_books = self.source_manager.get_available_books()
            
            return self.source_manager.format_selection_summary(self._cached_books, selected_books)
            
        except Exception as e:
            logger.error(f"Error updating selection summary: {e}")
            return f"❌ Error: {str(e)}"
    
    def validate_selection(self, selected_books: List[str]) -> Tuple[bool, str]:
        """
        Validate current selection
        
        Args:
            selected_books: List of selected book names
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not selected_books:
            return False, "Please select at least one book to search."
        
        try:
            if not self._cached_books:
                self._cached_books = self.source_manager.get_available_books()
            
            # Check if all selected books are available
            available_names = {book.name for book in self._cached_books if book.is_available}
            invalid_selections = [name for name in selected_books if name not in available_names]
            
            if invalid_selections:
                return False, f"Some selected books are not available: {', '.join(invalid_selections)}"
            
            return True, "Selection is valid"
            
        except Exception as e:
            logger.error(f"Error validating selection: {e}")
            return False, f"Validation error: {str(e)}"
    
    def get_selected_book_info(self, selected_books: List[str]) -> List[BookInfo]:
        """
        Get BookInfo objects for selected books
        
        Args:
            selected_books: List of selected book names
            
        Returns:
            List of BookInfo objects for selected books
        """
        if not self._cached_books:
            self._cached_books = self.source_manager.get_available_books()
        
        return [
            book for book in self._cached_books 
            if book.name in selected_books
        ]
    
    def create_ui_handlers(self) -> dict:
        """
        Create all UI event handlers
        
        Returns:
            Dictionary of handler functions for UI events
        """
        return {
            'refresh_sources': self.refresh_sources,
            'select_all': self.select_all_available,
            'clear_all': self.clear_all_selections, 
            'update_summary': self.update_selection_summary,
            'validate_selection': self.validate_selection
        }
    
    def calculate_smart_k(self, selected_books: List[str]) -> int:
        """
        Calculate intelligent k value based on selected books
        
        Args:
            selected_books: List of selected book names
            
        Returns:
            Optimal k value for search
        """
        if not selected_books:
            return 5  # Default
        
        try:
            selected_info = self.get_selected_book_info(selected_books)
            total_chapters = sum(book.successful_chapters for book in selected_info)
            
            # Smart k calculation: 
            # - Minimum 3 sources
            # - Maximum 15 sources
            # - Roughly 1 source per 10 chapters, but adapt to selection size
            if total_chapters <= 30:
                k = max(3, total_chapters // 5)
            elif total_chapters <= 100:
                k = max(5, total_chapters // 8)
            else:
                k = max(8, total_chapters // 12)
            
            k = min(k, 15)  # Cap at 15
            
            logger.debug(f"Smart k calculation: {total_chapters} chapters → k={k}")
            return k
            
        except Exception as e:
            logger.error(f"Error calculating smart k: {e}")
            return 5  # Safe default