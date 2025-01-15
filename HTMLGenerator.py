import re
import markdown
from bs4 import BeautifulSoup

def text_to_paragraphs(text):
    """Convert text with newlines to HTML paragraphs, handling both plain text and markdown.
    
    Args:
        text (str): Input text that may contain markdown or plain text
        
    Returns:
        str: HTML formatted text with proper paragraph tags
    """
    # First, check if the text appears to contain markdown
    markdown_indicators = ['#', '*', '_', '`', '[', '>', '-', '+']
    has_markdown = any(indicator in text for indicator in markdown_indicators)
    
    if has_markdown:
        # Convert markdown to HTML using the markdown library
        md = markdown.Markdown(extensions=['extra'])
        html_content = md.convert(text)
        
        # Clean up the HTML using BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Handle standalone text nodes by wrapping them in paragraphs
        for text_node in soup.find_all(text=True):
            if text_node.parent.name == 'body' or text_node.parent.name == '[document]':
                # Only wrap if it's not just whitespace
                if text_node.strip():
                    new_p = soup.new_tag('p')
                    text_node.wrap(new_p)
        
        return str(soup)
    else:
        # Handle plain text by splitting on newlines and wrapping in paragraph tags
        paragraphs = text.split('\n')
        return '\n'.join([f"<p>{p.strip()}</p>" for p in paragraphs if p.strip()])

def generate_html(book_title, book_summary, book_chapters, part_summaries={}):
    """Generate HTML content from book dictionary with markdown support.
    
    Args:
        book_dict (dict): Dictionary containing book content and structure
        
    Returns:
        str: Complete HTML document as string
    """
    # HTML template with styles
    html_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        @page {
            size: A4;
            margin: 2.5cm;
        }
        body {
            font-family: "Times New Roman", Times, serif;
            font-size: 12pt;
            line-height: 1.5;
            max-width: 21cm;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            font-size: 24pt;
            text-align: center;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        h2 {
            font-size: 18pt;
            margin-top: 30px;
        }
        .section-container {
            margin-bottom: 20px;
        }
        h3 {
            font-size: 14pt;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        h4 {
            font-size: 13pt;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        p {
            margin-bottom: 12pt;
            text-align: justify;
            orphans: 3;
            widows: 3;
            word-wrap: break-word;
            overflow-wrap: break-word;
            hyphens: auto;
            -webkit-hyphens: auto;
            -ms-hyphens: auto;
            page-break-inside: avoid;
        }
        .summary {
            margin: 20px 0;
            font-style: italic;
        }
        .chapter-summary {
            margin: 18px 0;
            font-style: italic;
            padding: 12px;
            background-color: #f8f8f8;
            border-left: 3px solid #333;
        }
        .page-break {
            page-break-after: always;
        }
        .chapter-container {
            margin-bottom: 40px;
        }
        .title-container {
            page-break-inside: avoid;
            margin-bottom: 30px;
        }
        hr {
            margin: 20px 0;
            border: none;
            border-top: 1px solid #000;
        }
        /* Additional styles for markdown elements */
        blockquote {
            margin: 15px 30px;
            padding-left: 15px;
            border-left: 3px solid #ccc;
        }
        code {
            font-family: monospace;
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        em {
            font-style: italic;
        }
        strong {
            font-weight: bold;
        }
    </style>
</head>
<body>
"""
    
    # Add title and main summary
    html = html_template
    html += "<div class='title-container'>\n"
    html += f"<h1>{book_title}</h1>\n"
    html += f"<div class='summary'>{text_to_paragraphs(book_summary)}</div>\n"
    html += "</div>\n"
    html += "<hr>\n"

    html += "<div class='page-break'></div>"

    if not part_summaries:
        # Add chapter summaries overview
        for chapter_title, chapter in book_chapters.items():
            html += f"<div class='chapter-summary'><h3>{chapter_title}</h3>\n"
            html += f"{text_to_paragraphs(chapter['summary']['text'])}</div>\n"
    else:
        for part_title, part in part_summaries.items():
            html += f"<div class='chapter-summary'><h3>{part_title}</h3>\n"
            html += f"{text_to_paragraphs(part['summary']['text'])}</div>\n"
    
    html += "<div class='page-break'></div>"

    # Process each chapter in detail
    for chapter_title, chapter in book_chapters.items():
        html += "<div class='chapter-container'>\n"
        html += f"<h2>{chapter_title}</h2>\n"
        if part_summaries:
            for part in part_summaries:
                html += f"<b>{part} - {' '.join(part_summaries[part])}</b>\n"
                html += f"{text_to_paragraphs(part_summaries[part])}</div>\n"
                html += "<hr>\n"

        html += "<hr>\n"
        
        # Process each section
        for section in chapter['content']:
            html += "<div class='section-container'>\n"
            html += f"<h3>{section['title']}</h3>\n"
            html += text_to_paragraphs(section['summary']['text']) + "\n"
            html += "</div>\n"

        html += "<div class='page-break'></div>"
            
        html += "</div>\n"
    
    html += "</body></html>"
    return html

def save_html(html_content, filename="book_summary.html"):
    """Save the generated HTML to a file
    
    Args:
        html_content (str): HTML content to save
        filename (str): Output filename
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)