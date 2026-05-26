import logging
import pytesseract
from PIL import Image
import io
import pdfplumber


class OCRProcessor:
    def __init__(self, lang='chi_sim+eng', enable_ocr=True):
        """
        初始化OCR处理器
        
        Args:
            lang: OCR识别语言，默认中文+英文
        """
        self.lang = lang
        self.enable_ocr = enable_ocr
        if not self.enable_ocr:
            self.available = False
            return

        # 检查tesseract是否可用
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logging.warning(f"Tesseract not found: {e}. OCR functionality will be limited.")
            self.available = False
        else:
            self.available = True

    def _clean_cell(self, cell):
        if cell is None:
            return ""
        return " ".join(str(cell).split()).replace("|", "\\|")

    def _table_to_markdown(self, table):
        rows = []
        for row in table or []:
            cleaned = [self._clean_cell(cell) for cell in row]
            if any(cleaned):
                rows.append(cleaned)

        if not rows:
            return ""

        width = max(len(row) for row in rows)
        rows = [row + [""] * (width - len(row)) for row in rows]

        header = rows[0]
        if not any(header):
            header = [f"Column {idx + 1}" for idx in range(width)]
            body = rows
        else:
            body = rows[1:]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for row in body:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def extract_tables_from_pdf(self, pdf_path):
        """
        Extract structured tables from a PDF and serialize them as Markdown.

        This handles digitally generated PDFs. Scanned tables still need OCR
        and are returned by process_pdf_with_ocr as image text.
        """
        tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        page_tables = page.extract_tables() or []
                    except Exception as e:
                        logging.warning(f"Table extraction failed on page {page_num}: {e}")
                        continue

                    for table_idx, table in enumerate(page_tables, 1):
                        markdown = self._table_to_markdown(table)
                        if markdown.strip():
                            tables.append(
                                {
                                    "text": markdown,
                                    "page": page_num,
                                    "index": table_idx,
                                    "type": "table",
                                }
                            )
        except Exception as e:
            logging.error(f"Error extracting PDF tables: {e}")

        return tables
    
    def extract_images_from_pdf(self, pdf_path):
        """
        从PDF中提取图片
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            图片数据列表
        """
        images = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_images = page.images
                    for img_idx, img in enumerate(page_images):
                        try:
                            img_data = img["stream"].get_data()
                            # 过滤过小的图片
                            if len(img_data) < 100:
                                continue
                            images.append({
                                "data": img_data,
                                "page": page_num,
                                "index": img_idx
                            })
                        except Exception as e:
                            logging.error(f"提取图片时出错: {e}")
                            continue
        except Exception as e:
            logging.error(f"处理PDF时出错: {e}")
        
        return images
    
    def ocr_image(self, image_data):
        """
        对图片进行OCR识别
        
        Args:
            image_data: 图片数据
            
        Returns:
            识别的文本
        """
        if not self.available:
            return ""
        
        try:
            # 将图片数据转换为PIL Image
            image = Image.open(io.BytesIO(image_data))
            # 进行OCR识别
            text = pytesseract.image_to_string(image, lang=self.lang)
            return text
        except Exception as e:
            logging.error(f"OCR识别时出错: {e}")
            return ""
    
    def process_pdf_with_ocr(self, pdf_path):
        """
        处理PDF文件，提取图片并进行OCR识别
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            识别的文本列表
        """
        images = self.extract_images_from_pdf(pdf_path)
        ocr_results = []
        
        for img_info in images:
            text = self.ocr_image(img_info["data"])
            if text.strip():
                ocr_results.append({
                    "text": text,
                    "page": img_info["page"],
                    "index": img_info["index"]
                })
        
        return ocr_results
