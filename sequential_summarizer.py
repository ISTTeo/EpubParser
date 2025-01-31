from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
import tiktoken
import json, os, pickle, requests
from util_summarizer import *

encoding = tiktoken.encoding_for_model("gpt-4")
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