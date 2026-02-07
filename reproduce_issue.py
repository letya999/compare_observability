
from typing import List, Dict, Any, Tuple
import re

# Mocking the PDF structure and classes
class MockPage:
    def __init__(self, rect_height: float, blocks: List[Tuple]):
        self.rect = type('obj', (object,), {'height': rect_height})
        self.blocks = blocks

    def get_text(self, option: str):
        if option == "blocks":
            return self.blocks
        return ""

class MockDoc:
    def __init__(self, pages: List[MockPage], metadata: Dict):
        self.pages = pages
        self.metadata = metadata

    def __iter__(self):
        return iter(self.pages)
    
    def __len__(self):
        return len(self.pages)
        
    def get_toc(self):
        return []

# Copying the relevant methods from PDFIngestor for testing
class PDFIngestorMock:
    def _extract_clean_content(
        self, doc, chapter_map: Dict[int, str], doc_title: str
    ) -> List[Dict[str, Any]]:
        """Extract text blocks, filtering headers/footers and noise."""
        items = []
        current_chapter = "Unknown"
        doc_author = doc.metadata.get("author", "")

        sorted_chapters = sorted(chapter_map.keys())
        
        for page_num, page in enumerate(doc):
            # Update chapter
            if page_num in chapter_map:
                current_chapter = chapter_map[page_num]
            elif sorted_chapters:
                valid_chaps = [p for p in sorted_chapters if p <= page_num]
                if valid_chaps:
                    current_chapter = chapter_map[valid_chaps[-1]]

            blocks = page.get_text("blocks")
            page_height = page.rect.height
            
            top_margin = page_height * 0.08
            bottom_margin = page_height * 0.92

            for b in blocks:
                x0, y0, x1, y1, text, _, _ = b
                text = text.strip()
                if not text:
                    continue

                # --- 1. Geometry Filter (Header/Footer) ---
                is_header_footer = y1 < top_margin or y0 > bottom_margin
                
                # --- 2. Content Heuristics ---
                # Detect page numbers (digits only, or "Page X")
                is_page_num = re.match(r'^(page\s?)?\d+$', text.lower())
                
                # Detect URL/Domain patterns common in headers
                is_url = bool(re.search(r'(www\.|http:|https:|\.com|\.ru|\.org)', text.lower()))
    
                # Detect Title/Author repetition (simple header check)
                is_title_noise = (len(text) < 100) and (
                    doc_title.lower() in text.lower() or 
                    (doc_author and doc_author.lower() in text.lower())
                )
    
                if is_header_footer and (is_page_num or is_title_noise or is_url):
                    continue

                items.append({
                    "text": text,
                    "page": page_num + 1,
                    "chapter": current_chapter,
                    "bbox": (x0, y0, x1, y1)
                })
        
        return items

    def _merge_content_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge small blocks into paragraphs to improve flow and avoid mid-sentence cuts."""
        if not items:
            return []
            
        merged = []
        current = items[0]
        
        for next_item in items[1:]:
            # Check for sentence ending punctuation or clear structural breaks
            text = current["text"].strip()
            # If text ends with hyphen, remove it and merge without space
            is_hyphenated = text.endswith("-") or text.endswith("–") # check widely
            
            # Simple heuristic: Merge if same page/chapter and not clearly a new paragraph
            # (Note: indent detection is hard without more logic, but "ends in punctuation" is a good proxy for "don't merge")
            
            has_terminator = text[-1] in ".!?" if text else True
            
            should_merge = (
                current["page"] == next_item["page"] and
                current["chapter"] == next_item["chapter"] and
                (not has_terminator or is_hyphenated)
            )
            
            if should_merge:
                if is_hyphenated:
                     # Remove hyphen and join directly
                    current["text"] = current["text"].rstrip("-–") + next_item["text"]
                else:
                    current["text"] += " " + next_item["text"]
                
                # Update bbox (union)
                c_box = current["bbox"]
                n_box = next_item["bbox"]
                current["bbox"] = (
                    min(c_box[0], n_box[0]),
                    min(c_box[1], n_box[1]),
                    max(c_box[2], n_box[2]),
                    max(c_box[3], n_box[3])
                )
            else:
                merged.append(current)
                current = next_item
        
        merged.append(current)
        return merged

# Setup Data
# Page height 800. Margin 8% = 64.
# Header at y1=50 (inside top margin)
# Text at y0=70 (safe)

header_text = "100 лучших книг всех времен: www.100bestbooks.ru"
page1_blocks = [
    (10, 70, 500, 100, "– Свистнуто, не спорю, – снисходительно заметил Коровьев", 0, 0),
    (10, 110, 500, 140, "– Я ведь не регент, –", 1, 0)
]
page2_blocks = [
    (10, 10, 500, 50, header_text, 0, 0), # Header - should be filtered but currently isn't 
    (10, 70, 500, 100, "с достоинством заметил Коровьев", 1, 0),
    (10, 110, 500, 140, "Тогда прекратился свист швейцара", 2, 0)
]

doc = MockDoc(
    pages=[
        MockPage(800, page1_blocks),
        MockPage(800, page2_blocks)
    ],
    metadata={"title": "Master and Margarita", "author": "Bulgakov"}
)

ingestor = PDFIngestorMock()
cleaned_items = ingestor._extract_clean_content(doc, {}, "Master and Margarita")
cleaned_items = ingestor._merge_content_items(cleaned_items)

print("--- Extracted Items ---")
for item in cleaned_items:
    print(f"[{item['page']}] {item['text']}")

