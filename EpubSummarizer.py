from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
import tiktoken
import json, os, pickle, requests


encoding = tiktoken.encoding_for_model("gpt-4")

def summarize_text_OPENAI(messages, openai_key, model):
    client = OpenAI(api_key=openai_key)

    response = client.chat.completions.create(
        model=model, 
        messages = messages,
    )
    summary = response.choices[0].message.content.strip()
    return summary, messages

def summarize_text_LMSTUDIO(messages, tries=5):
    
    payload = {
        "model": "phi-4@q6_k", 
        "messages": messages,
    }
    
    

    while tries > 0:
        try:
            response = requests.post(
                "http://localhost:24236/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=15
            )
            if response.status_code == 200:
                result = response.json()
                summary = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return summary, messages
            else:
                tries -=1 
        except:
            tries -=1

    raise Exception(f"Tries ran out --> {messages}")



def summarize_text(messages, openai_key=None, model="gpt-4o-mini"):
    if openai_key is not None:
        return summarize_text_OPENAI(messages, openai_key, model)
        
    return summarize_text_LMSTUDIO(messages)

def deep_copy_dict(dictionary):
    # Base case - if input is not a dictionary
    if not isinstance(dictionary, dict):
        # If it's a list, deep copy each element
        if isinstance(dictionary, list):
            return [deep_copy_dict(item) for item in dictionary]
        # If it's any other type, return as is (assuming immutable)
        return dictionary
    
    # Create new dictionary for the copy
    copy = {}
    
    # Iterate through key-value pairs
    for key, value in dictionary.items():
        # Recursively copy nested dictionaries/lists
        copy[key] = deep_copy_dict(value)
        
    return copy


class EPUBSummarizerSequential:
    def __init__(self, title, organized_book, openai_key=None):
        self.book_title = title
        self.based_organized_book = organized_book
        self.A_summarized_sections = {}
        self.B_summarized_chapters = {}
        self.C_book_summary = ""
        self.D_summarized_chapters_with_book_context = {}
        self.E_summarized_sections_with_book_and_chapter_context = {}
        self.openai_key = openai_key
        self.STYLING_INSTRUCTION = "Avoid any styling asides from bold, italics, bullet-points or enumerations. Avoid any headers or other hierarchical elements. Return markdown formatting without any tags around it."


    def get_fingerprint(self):

        if self.openai_key is None:
            model= "ownModel"
        else:
            model="OpenAI"

        return "_".join(self.book_title.split()) + "__" + model

    def A_summarize_sections_for_chapters(self, function_messages_from_text, keys_to_skip):
        #print(f"Skipping: {keys_to_skip}\n")
        for chapter_title, chapter in self.based_organized_book.items():
            if chapter_title not in keys_to_skip:
                #print(chapter_title)
                groups = []
                aux = []
                previous_title = ""

                if len(chapter) == 1:
                    #print(chapter)
                    #print('skipping')
                    continue

                for i, entry in enumerate(chapter):
                    if (i+1)%10 == 0:
                        print(f"\t{i+1}/{len(chapter)}")
                    if entry[0] == 'title':
                        if previous_title:
                            
                            text = "##" + previous_title + "\n".join(aux)
                            
                            messages = function_messages_from_text(text)

                            summary_text, messages = summarize_text(messages, openai_key=self.openai_key)

                            strategy = "Collect paragraphs within subsection (between titles) and summarize those. " + str(messages) 
                            
                            token_counts = [len(encoding.encode(pg)) for pg in aux]

                            section = {
                                "title":previous_title,
                                "original_text": (aux, token_counts),
                                "summary": {
                                    "strategy": strategy,
                                    "text": summary_text
                                }
                            }
                            
                            groups.append(section)
                            previous_title = entry[1]
                            aux = []
                        else:
                            previous_title = entry[1]
                
                    if entry[0] == 'paragraph':
                        text = entry[1]
                        aux.append(text)

                if aux:
                    text = "##" + previous_title + "\n".join(aux)
                                    
                    
                    token_counts = [len(encoding.encode(pg)) for pg in aux]

                    messages = function_messages_from_text(text)
                    
                    summary_text, messages = summarize_text(messages, openai_key=self.openai_key)

                    strategy = "Collect paragraphs within subsection (between titles) and summarize those. " + str(messages) 

                    section = {
                        "title":previous_title,
                        "original_text": (aux, token_counts),
                        "summary": {
                            "strategy": strategy,
                            "text": summary_text
                        }
                    }
                    
                    groups.append(section)

                self.A_summarized_sections[chapter_title] = groups

        pickle_file = "A_summarized_sections__" + self.get_fingerprint() 
        with open(pickle_file, "wb") as f:
            pickle.dump(self.A_summarized_sections, f)

    def B_summarize_chapter_from_sections_summaries(self):
        self.B_summarized_chapters = deep_copy_dict(self.A_summarized_sections)

        for c_title, chapter in self.B_summarized_chapters.items():
            #print(c_title)

            internal_text = ""
            
            if type(chapter) == dict and 'content' in chapter.keys():
                chapter = chapter['content']
            
            for section in chapter:
                section_title = section['title']
                summary_text = section['summary']['text']
                
                internal_text += f"\n##{section_title}" + "\n" + summary_text

            instruction = "Summarize the following text composed of summaries of each subsection of the chapter."
            instruction += f"\nThe chapter is '{c_title}' and its contents: {internal_text}"            
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure." + self.STYLING_INSTRUCTION},
                {"role": "user", "content": instruction}
            ]

            chapter_summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key, model="gpt-4o")
            
            strategy = "Take the collected summaries for subsections, the previous book summary and then summarize using: " + str(messages)

            chapter = {"summary": 
                        {"strategy": strategy,
                            "text": chapter_summary_text},
                        "content": chapter}

            self.B_summarized_chapters[c_title] = chapter

        pickle_file = "B_summarized_chapters" + "__" + self.get_fingerprint() 
        with open(pickle_file, "wb") as f:
            pickle.dump(self.B_summarized_chapters, f)

    def C_summarize_book_from_chapters(self):
        book_title = self.book_title

        concatenated = ""
        for title, chapter in self.B_summarized_chapters.items():
            #print(title)
            concatenated += f"## Chapter '{title}'" + "\n"
            concatenated += f"{chapter['summary']['text']}"
            concatenated += "\n\n"

        messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure. " + self.STYLING_INSTRUCTION},
                {"role": "user", "content": f"Create a concluding summary (in the number of appropriate paragraphs) of the book {book_title} from the summaries of its chapters: {concatenated}"}
            ]

        book_summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key, model="gpt-4o")
            
        self.C_book_summary = book_summary_text

        pickle_file = "C_book_summary" + "__" + self.get_fingerprint() 
        with open(pickle_file, "wb") as f:
            pickle.dump(self.C_book_summary, f)

    
    def D_summarize_chapter_from_sections_summaries_and_book_summary(self):
        self.D_summarized_chapters_with_book_context = deep_copy_dict(self.B_summarized_chapters)
        book_summary = self.C_book_summary

        for c_title, chapter in self.D_summarized_chapters_with_book_context.items():
            #print(c_title)

            internal_text = ""
            if type(chapter) == dict and 'content' in chapter.keys():
                chapter = chapter['content']
            
            for section in chapter:
                section_title = section['title']
                summary_text = section['summary']['text']
                
                internal_text += f"\n##{section_title}" + "\n" + summary_text

            instruction = "Summarize the following text composed of summaries of each subsection of the chapter."
            instruction += f"Take into account the broader context of the book while still focusing on the chapter, this context comes from its summary:\n {book_summary}"
            instruction += f"\nThe chapter is '{c_title}' and its contents: {internal_text}"            
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure. Return markdown formatting without any tags around it."},
                {"role": "user", "content": instruction}
            ]

            chapter_summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key, model="gpt-4o")
            
            strategy = "Take the collected summaries for subsections, the previous book summary and then summarize using: " + str(messages)

            chapter = {"summary": 
                        {"strategy": strategy,
                            "text": chapter_summary_text},
                        "content": chapter}

            self.D_summarized_chapters_with_book_context[c_title] = chapter

        pickle_file = "D_summarized_chapters_with_book_context" + "__" + self.get_fingerprint() 
        with open(pickle_file, "wb") as f:
            pickle.dump(self.D_summarized_chapters_with_book_context, f)
        
    def E_summarize_sections_with_book_and_chapter_summaries(self):
        self.E_summarized_sections_with_book_and_chapter_context = deep_copy_dict(self.D_summarized_chapters_with_book_context)

        book_summary = self.C_book_summary

        for c_title, chapter in self.E_summarized_sections_with_book_and_chapter_context.items():
            chapter_summary = chapter['summary']['text']
            #print(c_title)
            section_summary_text = ""

            if len(chapter) == 1 and chapter[0][0] == 'title':
                #print('skipping')
                continue
            
            for i, section in enumerate(chapter['content']):     
                internal_text = "\n".join(section['original_text'][0])
                title = section['title']
                #print(f"\t{title}")
                
                instruction = "Summarize the following text composed of paragraphs of a section of a chapter."
                instruction += f"Take into account the broader context of the book while still focusing on the section mainly." 
                instruction += f"\nThis context comes from its summary:\n {book_summary}"
                instruction += f"\nAnd from the chapter summary:\n {chapter_summary}"
                if section_summary_text:
                    instruction += f"\n Also take into account the previous section in order for it to flow properly: {section_summary_text}"
                else:
                    instruction += "This is the first section of this chapter"
                instruction += "Take into account that each the resulting summary will be used preceded by it's title so avoid mentioning it."
                instruction += "It will also be grouped within it's chapter surrounded by the neighbouring section summaries, this should be considered to make sure the text flows properly."
                instruction += f"\nThe section to summarize is '{title}' and its contents: {internal_text}"            

                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure. " + self.STYLING_INSTRUCTION},
                {"role": "user", "content": instruction}
                ]
            
                section_summary_text, messages = summarize_text(messages=messages, openai_key=self.openai_key)
                
                strategy = "Take the collected summaries for subsections, the previous book summary and then summarize using: " + str(messages)
            
                self.E_summarized_sections_with_book_and_chapter_context[c_title]['content'][i]['summary'] = {"strategy": strategy, "text": section_summary_text}
        
        pickle_file = "E_summarized_sections_with_book_and_chapter_context" + "__" + self.get_fingerprint() 
        with open(pickle_file, "wb") as f:
            pickle.dump(self.E_summarized_sections_with_book_and_chapter_context, f)
        
        def F_summarize_by_selection(self, selection_dict):
            result = {}
            for k, chapter_keys in selection_dict.items():
                concatenated = ""
                for title in chapter_keys:
                    chap_summary = self.E_summarized_sections_with_book_and_chapter_context['summary']['text']
                    #print(title)
                    concatenated += f"## Chapter '{title}'" + "\n"
                    concatenated += f"{chap_summary}"
                    concatenated += "\n\n"

                instruction = "Summarize the following text composed of summaries of chapters of a book."
                instruction += f"\nThis context comes from the book {self.book_title} with its summary:\n {book_summary}"
                instruction += f"Create a summary for this group of chapters named '{k}' with the contents of the summaries being: {concatenated}"
                messages=[
                            {"role": "system", "content": "You are a helpful assistant that summarizes paragraphs from books. Use paragraphs to break logically and to enable better structure. Return markdown formatting without any tags around it."},
                            {"role": "user", "content": instruction}
                    ]

                part_summary, messages = summarize_text(messages=messages, openai_key=self.openai_key, model="gpt-4o")
                result[k] = part_summary

            pickle_file = "F_summarize_by_selection" + "__" + self.get_fingerprint() 
            saved = {
                "selection_dict": selection_dict,
                "result": result
            }

            with open(pickle_file, "wb") as f:
                pickle.dump(saved, f)

            return result


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
        
        titles_to_ignore = [t.lower() for t in titles_to_ignore]

        if node.get('type') == "title" and node.get('content').lower() in titles_to_ignore:
            indent_str = '\t' * indent
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
            try:
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
            
            except Exception as e:
                print(f"Error processing section: {e}")
                # Continue with next section even if one fails
                continue
            
        
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