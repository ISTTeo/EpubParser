from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
import tiktoken
import json, os, pickle, requests
from util_summarizer import *

encoding = tiktoken.encoding_for_model("gpt-4")

class EPUBSummarizerHierarchical:

    def __init__(self, title: str, organized_book: List[Dict], openai_key: Optional[str] = None, use_cache: bool = True):
        self.book_title = title
        self.based_organized_book = organized_book
        self.summarized_book = {}
        self.contextually_summarized_book = {}
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.openai_key = openai_key
        self.use_cache = use_cache
        self.cache_dir = "summarizer_cache"
        self.global_summary = ""
        self.STYLING_INSTRUCTION = "Avoid any styling asides from bold, italics, bullet-points or enumerations. Avoid any headers or other hierarchical elements. Do not reference the section title in the summary, just summarize the contents encapsulated within it. Return markdown formatting without any tags around it."
        self._initialize_cache()

    def _initialize_cache(self):
        """Initialize cache directory if it doesn't exist"""
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def get_fingerprint(self) -> str:
        mode = "ownModel" if self.openai_key is None else "OpenAI"
        return "_".join(self.book_title.split()) + "__" + mode

    def _get_cache_path(self, node_id: str) -> str:
        """Get the cache file path for a specific node"""
        sanitized_id = "".join(c for c in node_id if c.isalnum() or c in ('-', '_'))
        return os.path.join(self.cache_dir, f"{self.get_fingerprint()}__{sanitized_id}.pkl")

    def _generate_node_id(self, node: Dict, context: str) -> str:
        """Generate a unique identifier for a node based on its content and context"""
        content = node.get('content', '')
        node_type = node.get('type', '')
        subtype = node.get('subtype', '')

        #print(node)

        return f"{node_type}_{subtype}_{hash(content + context)}"

    def save_to_cache(self, node_id: str, data: Dict):
        """Save node data to cache"""
        if not self.use_cache:
            return
        
        cache_path = self._get_cache_path(node_id)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Warning: Failed to save to cache: {e}")

    def load_from_cache(self, node_id: str) -> Optional[Dict]:
        """Load node data from cache if available"""
        if not self.use_cache:
            return None
            
        cache_path = self._get_cache_path(node_id)
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load from cache: {e}")
        return None

    def save_state(self, stage: str):
        """Save current state to a pickle file"""
        pickle_file = f"{stage}__{self.get_fingerprint()}"
        with open(pickle_file, "wb") as f:
            pickle.dump(self.summarized_book, f)

    def summarize_content(self, node: Dict, context: str = "", indent: int = 0, titles_to_ignore: List[str] = None) -> Optional[Dict]:
        """Recursively summarize content at each level while maintaining hierarchical structure"""
        if titles_to_ignore is None:
            titles_to_ignore = []

        if not isinstance(node, dict):
            return node
        
        if type(node['content']) == list:
                node['content'] = " ".join(node['content'])

        titles_to_ignore = [t.lower() for t in titles_to_ignore]

        if node.get('type') == "title":
            indent_str = '\t' * indent
            if node.get('content').lower() in titles_to_ignore:
                print(f"{indent_str}Skipping -> {node['content']}")
                return None
            
        # Generate unique ID for this node
        node_id = self._generate_node_id(node, context)
        
        # Try to load from cache first
        cached_result = self.load_from_cache(node_id)
        if cached_result is not None:
            return cached_result

        result = {'type': node['type']}
        if 'subtype' in node:
            result['subtype'] = node['subtype']
        
        # Handle content directly if present
        if 'content' in node:
            result['content'] = node['content']
            if node['type'] == 'title':
                indent_str = '\t' * indent
                print(f"{indent_str}{node['content']}")

        # Process children if present
        if 'children' in node:
            result['children'] = []
            ordered_children = []
            has_non_paragraph = False
            
            # First process all children and maintain order
            for child in node['children']:
                summarized_child = self.summarize_content(
                    child, 
                    context, 
                    indent + 1, 
                    titles_to_ignore=titles_to_ignore
                )
                if summarized_child is not None:
                    result['children'].append(summarized_child)
                    if summarized_child['type'] != 'paragraph':
                        has_non_paragraph = True
                    ordered_children.append({
                        'type': summarized_child['type'],
                        'content': summarized_child.get('content', ''),
                        'summary': summarized_child.get('summary', {}).get('text', '')
                    })

            # Generate summary based on children types
            if ordered_children:
                # Case 1: Only paragraph children - summarize all paragraphs together in order
                if not has_non_paragraph:
                    ordered_text = ' '.join(child['content'] for child in ordered_children)
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure. " + self.STYLING_INSTRUCTION},
                        {"role": "user", "content": f"Summarize the following sequence of paragraphs in the context of {context}, maintaining the logical flow of ideas:\n\n{ordered_text}"}
                    ]
                    try:
                        summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key)
                        result['summary'] = {
                            "strategy": f"Summarized ordered sequence of paragraphs with context: {context}",
                            "text": summary_text
                        }
                    except Exception as e:
                        print(f"Warning: Failed to generate summary: {e}")
                        result['summary'] = {
                            "strategy": "Failed to generate summary - using concatenated paragraphs",
                            "text": ordered_text
                        }
                
                # Case 2: Mixed children or only non-paragraph children
                else:
                    # Build ordered text maintaining document flow
                    ordered_text = ""
                    for child in ordered_children:
                        if child['type'] == 'paragraph':
                            ordered_text += child['content'] + "\n\n"
                        else:
                            ordered_text += child['summary'] + "\n\n"
                    
                    cnt_str = f"\nThe content of this is: {result['content']}" if result.get('content') else ""
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure. " + self.STYLING_INSTRUCTION},
                        {"role": "user", "content": f"Summarize the following content in the context of {context}, maintaining the logical flow and sequence of ideas:\n\n{ordered_text}{cnt_str}"}
                    ]
                    try:
                        summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key)
                        result['summary'] = {
                            "strategy": f"Summarized from ordered sequence of children summaries and paragraphs with context: {context}",
                            "text": summary_text
                        }
                    except Exception as e:
                        print(f"Warning: Failed to generate summary: {e}")
                        result['summary'] = {
                            "strategy": "Failed to generate summary - using concatenated ordered content",
                            "text": ordered_text
                        }

        # Save result to cache
        self.save_to_cache(node_id, result)
        return result

    def process_book(self, titles_to_ignore=[]):
        """Process the entire book structure"""
        print("Starting book processing...")
        
        self.summarized_book = []
        
        # Process each top-level section
        for section in self.based_organized_book:
            summarized_section = self.summarize_content(
                section,
                f"Book: {self.book_title}",
                titles_to_ignore=titles_to_ignore
            )
            if summarized_section is None:
                continue
            else:
                self.summarized_book.append(summarized_section)
                
                # Save state after each major section
                self.save_state("full_book_processing")

            
        
        print("Book processing complete.")
        return self.summarized_book

    def get_summary_by_level(self, level: str) -> Dict[str, str]:
        """Extract summaries for a specific hierarchical level"""
        def extract_summaries(node: Dict, level_type: str, results: Dict):
            if not isinstance(node, dict):
                return
                
            if 'subtype' in node and node['subtype'] == level_type and 'summary' in node:
                title = node.get('content', 'Untitled')
                results[title] = node['summary']['text']
                
            if 'children' in node:
                for child in node['children']:
                    extract_summaries(child, level_type, results)
        
        summaries = {}
        for section in self.summarized_book:
            extract_summaries(section, level, summaries)
            
        return summaries
    
    def create_global_summary(self, subtypes_to_use=None) -> str:
        """
        Create a global summary of the book using summaries from specified subtypes.
        
        Args:
            subtypes_to_use (List[str], optional): List of subtypes to consider for global summary.
                If None, uses all available summaries.
                Common subtypes might include ['chapter', 'section', 'subsection']
        
        Returns:
            str: Global summary of the book
        """
        def collect_summaries(node: Dict, subtypes: List[str], collected_summaries: List[Dict]) -> None:
            """Recursively collect summaries from specified subtypes."""
            if not isinstance(node, dict):
                return
                
            # If node has a summary and matches subtype criteria, collect it
            if 'summary' in node and 'subtype' in node:
                if subtypes is None or node['subtype'] in subtypes:
                    collected_summaries.append({
                        'title': node.get('content', 'Untitled'),
                        'subtype': node.get('subtype', 'unknown'),
                        'summary': node['summary']['text']
                    })
                    
            # Process children
            if 'children' in node:
                for child in node['children']:
                    collect_summaries(child, subtypes, collected_summaries)
        
        # Collect relevant summaries
        summaries = []
        for section in self.summarized_book:
            collect_summaries(section, subtypes_to_use, summaries)
        
        if not summaries:
            raise ValueError("No summaries found for the specified subtypes")
        
        # Organize summaries by subtype for better context
        organized_text = ""
        for summary in summaries:
            organized_text += f"\n## {summary['subtype'].title()}: {summary['title']}\n"
            organized_text += summary['summary']
            organized_text += "\n"
        
        # Generate global summary using the organized summaries
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates comprehensive book summaries. Use clear paragraphs to break down main themes and key points." + self.STYLING_INSTRUCTION},
            {"role": "user", "content": f"Create a comprehensive summary of the book '{self.book_title}' based on these component summaries:\n\n{organized_text}"}
        ]
        
        try:
            global_summary, messages = summarize_text(messages=messages, openai_key=self.openai_key)
            
            # Cache the global summary
            cache_data = {
                "subtypes_used": subtypes_to_use,
                "summary": global_summary,
            }
            self.save_to_cache("global_summary", cache_data)
            self.global_summary = global_summary
            return global_summary
            
        except Exception as e:
            print(f"Warning: Failed to generate global summary: {e}")
            # Fallback to concatenating summaries if AI summarization fails
            return f"# Summary of {self.book_title}\n\n{organized_text}"

    def get_available_subtypes(self) -> List[str]:
        """
        Get a list of all available subtypes in the book structure.
        Useful for selecting which subtypes to use in global summary.
        
        Returns:
            List[str]: List of unique subtypes found in the book
        """
        subtypes = set()
        
        def collect_subtypes(node: Dict) -> None:
            if not isinstance(node, dict):
                return
                
            if 'subtype' in node:
                subtypes.add(node['subtype'])
                
            if 'children' in node:
                for child in node['children']:
                    collect_subtypes(child)
        
        for section in self.summarized_book:
            collect_subtypes(section)
            
        return sorted(list(subtypes))
    
    def create_contextual_summaries(self):
        """
        Creates summaries in a top-down approach, where each level uses the context
        from all levels above it to create more cohesive summaries.
        """
        def summarize_with_context(node: Dict, context_summaries: Dict[str, str], indent: int = 0) -> Dict:
            """
            Recursively summarize nodes using context from higher levels.
            
            Args:
                node: Current node to summarize
                context_summaries: Dictionary of higher-level summaries (e.g., {'book': '...', 'chapter': '...'})
                indent: Current indentation level for printing
            """
            if not isinstance(node, dict):
                return node

            result = node.copy()
            
            # Print current node being processed
            indent_str = '\t' * indent
            if 'content' in node and node.get('type') == 'title':
                print(f"{indent_str}Processing: {node['content']}")

            # Build context string from higher levels
            context_text = "\n\n".join([f"## {level.title()} Context:\n{summary}" 
                                    for level, summary in context_summaries.items()])

            # Process children if present
            if 'children' in node:
                result['children'] = []
                ordered_children = []
                has_non_paragraph = False
                
                # Add current level context for children
                child_context = context_summaries.copy()
                if node.get('subtype'):
                    current_level_summary = node.get('summary', {}).get('text', '')
                    child_context[node['subtype']] = current_level_summary
                
                # Process each child
                for child in node['children']:
                    summarized_child = summarize_with_context(
                        child,
                        child_context,
                        indent + 1
                    )
                    if summarized_child is not None:
                        result['children'].append(summarized_child)
                        if summarized_child['type'] != 'paragraph':
                            has_non_paragraph = True
                        ordered_children.append({
                            'type': summarized_child['type'],
                            'content': summarized_child.get('content', ''),
                            'summary': summarized_child.get('summary', {}).get('text', '')
                        })

                # Generate summary based on children types
                if ordered_children:
                    # Case 1: Only paragraph children - summarize all paragraphs together in order
                    if not has_non_paragraph:
                        ordered_text = ' '.join(child['content'] for child in ordered_children)
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant that creates contextual summaries. Use the provided context to create cohesive summaries that fit within the larger narrative." + self.STYLING_INSTRUCTION},
                            {"role": "user", "content": f"""Create a summary that fits within this broader context:

    {context_text}

    Content to summarize (a sequence of paragraphs):

    {ordered_text}

    Note: This is a {node.get('subtype', 'section')} within the larger work. Maintain the logical flow while making the summary cohesive with the provided context."""}
                        ]
                        
                        try:
                            summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key)
                            result['summary'] = {
                                "strategy": f"Contextual summary of ordered paragraphs using higher-level contexts",
                                "text": summary_text
                            }
                        except Exception as e:
                            print(f"Warning: Failed to generate contextual summary: {e}")
                            result['summary'] = node.get('summary', {
                                "strategy": "Failed to generate new summary - using original",
                                "text": ordered_text
                            })
                    
                    # Case 2: Mixed children or only non-paragraph children
                    else:
                        # Build ordered text maintaining document flow
                        ordered_text = ""
                        for child in ordered_children:
                            if child['type'] == 'paragraph':
                                ordered_text += child['content'] + "\n\n"
                            else:
                                ordered_text += child['summary'] + "\n\n"
                        
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant that creates contextual summaries. Use the provided context to create cohesive summaries that fit within the larger narrative." + self.STYLING_INSTRUCTION},
                            {"role": "user", "content": f"""Create a summary that fits within this broader context:

    {context_text}

    Content to summarize (maintaining original sequence):

    {ordered_text}

    Note: This is a {node.get('subtype', 'section')} within the larger work. Maintain the logical flow while making the summary cohesive with the provided context."""}
                        ]
                        
                        try:
                            summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key)
                            result['summary'] = {
                                "strategy": f"Contextual summary of mixed content using higher-level contexts",
                                "text": summary_text
                            }
                        except Exception as e:
                            print(f"Warning: Failed to generate contextual summary: {e}")
                            result['summary'] = node.get('summary', {
                                "strategy": "Failed to generate new summary - using original",
                                "text": ordered_text
                            })

            return result

        # Start with global summary if it doesn't exist
        if not self.global_summary:
            self.create_global_summary()

        # Initialize with book-level context
        context_summaries = {'book': self.global_summary}
        
        # Process entire book with context
        new_summarized_book = []
        for section in self.summarized_book:
            new_section = summarize_with_context(section, context_summaries)
            new_summarized_book.append(new_section)
        
        # Update the book with new contextual summaries
        self.contextually_summarized_book = new_summarized_book
        
        # Save the contextual state
        self.save_state("contextual_summaries")
        
        return self.contextually_summarized_book