from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
from tqdm import tqdm
import tiktoken
import requests
import json
encoding = tiktoken.encoding_for_model("gpt-4")

class EPUBProcessor:
    """Process EPUB files into structured content with proper handling of chapters, 
    footnotes, and references."""
    
    def __init__(self, epub_path: str):
        """Initialize the processor with an EPUB file path.
        
        Args:
            epub_path (str): Path to the EPUB file
        """
        self.epub_path = epub_path
        self.book = None
        self.chapters = []
        self.chapters_html = []
        self.book_by_chapters = {}
        self.notes_by_chapter = {}
        self.parsed_chapters = []
        self.organized_book = {}

        self.C_function__set_section_types_by_classes_and_clean_text = None
    
    def _extract_chapters(self) -> None:
        """Extract chapters from the EPUB file."""
        item_label_number = {
            name: getattr(ebooklib, name) 
            for name in dir(ebooklib) if name.startswith('ITEM_')
        }
        
        self.chapters = [
            item for item in self.book.items 
            if item.get_type() == item_label_number["ITEM_DOCUMENT"]
        ]
    
    def _parse_chapters(self) -> None:
        """Parse chapters into BeautifulSoup objects."""
        self.chapters_html = []
        for chapter in self.chapters:
            content = chapter.get_content()
            parsed_content = BeautifulSoup(content, 'html.parser')
            text = parsed_content.get_text()
            
            if bool(text.strip()):
                self.chapters_html.append(parsed_content)
    
    def B_parse_chapters(self) -> List[List[Tuple[str, Any]]]:
        """Process all chapters and extract their elements.
        
        Returns:
            List[List[Tuple[str, Any]]]: List of processed chapters
        """
        self.parsed_chapters = [self.extract_elements(chapter_soup) for chapter_soup in self.chapters_html]
        return self.parsed_chapters

    def A_load_epub(self) -> None:
        """Load and parse the EPUB file."""
        try:
            self.book = epub.read_epub(self.epub_path)
            self._extract_chapters()
            self._parse_chapters()

        except Exception as e:
            raise Exception(f"Error loading EPUB file: {str(e)}")
        
    def extract_elements(self, chapter_soup: BeautifulSoup) -> List[Tuple[str, Any]]:
        """Extract elements from a chapter in sequential order, including classes and IDs.
        
        Args:
            chapter_soup (BeautifulSoup): BeautifulSoup object of chapter content
            
        Returns:
            List[Tuple[str, Any]]: List of tuples containing element type, content, classes, and IDs
        """
        elements = []
        for element in chapter_soup.find_all(['h1', 'h2', 'p', 'a', 'img', 'blockquote']):
            # Determine element type
            if element.name in ['h1', 'h2']:
                element_type = 'title'
            elif element.name == 'p':
                element_type = 'paragraph'
            elif element.name == 'a' and element.get('class') and any('Reference' in c for c in element['class']):
                element_type = 'note'
            elif element.name == 'img':
                element_type = 'media'
            elif element.name == 'blockquote':
                element_type = 'quote'
            else:
                continue
    
            # Fetch classes and IDs
            element_classes = element.get('class', [])  # Returns a list of classes or an empty list
            element_id = element.get('id', None)       # Returns the ID or None
    
            # Append element type, content, classes, and ID
            elements.append((element_type, element, element_classes, element_id))
        
        return elements
    
    def display_parsed_chapter(self):
        chapters = ""
        for i, c in enumerate(self.parsed_chapters):
            chapters += f"\n[{i}] ({c[0][0]}) -> {c[0][1]}"

        print(chapters)

    def check_parsed_chapters_classes(self):
        classes = {}

        for cI, c in enumerate(self.parsed_chapters):
            for sI, s in enumerate(c):
                for cla in s[2]:
                    identifier = (cI, sI)
                    if cla in classes.keys():
                        classes[cla].append(identifier)
                    else:
                        classes[cla] = [identifier]
        for k in classes:
            print(f"{k} -> {len(classes[k])}")

    def D_set_section_types_by_classes_and_clean_text(self):
        if self.C_function__set_section_types_by_classes_and_clean_text is not None:
            self.C_function__set_section_types_by_classes_and_clean_text()
        else:
            for cI, c in enumerate(self.parsed_chapters):
                for sI, s in enumerate(c):
                    self.parsed_chapters[cI][sI] = (s[0], s[1].get_text())

    def E_organize_book(self):     
        for chapter_sections in self.parsed_chapters:
            result = []
            last_type = None
            combined_title = ""
            first_title = ""

            for section_type, content in chapter_sections:
                if section_type == 'title':
                    if last_type == 'title':
                        # Combine with previous title
                        combined_title += " - " + content
                    else:
                        # Start new title
                        if combined_title:
                            result.append(('title', combined_title))
                        combined_title = content
                    
                    if not first_title:
                        first_title = combined_title
                else:
                    # Non-title section
                    if combined_title:
                        result.append(('title', combined_title))
                        combined_title = ""
                    result.append((section_type, content))
                last_type = section_type
            
            # Add any remaining title
            if combined_title:
                result.append(('title', combined_title))
            
            self.organized_book[first_title] = result
        
        for chapter_title, chapter_sections in self.organized_book.items():
            if (len(chapter_sections) == 1 and chapter_sections[0][0] == 'title'):
                self.organized_book[chapter_title] = self.organized_book[chapter_title][0][1]
        
        return self.organized_book
