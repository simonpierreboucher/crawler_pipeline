import os
import re
import logging
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

import requests
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import pypdf
import numpy as np
import cv2
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

# Charger les variables d'environnement depuis un fichier .env
load_dotenv()

# Configuration de Tesseract si nécessaire
# Décommentez et ajustez le chemin ci-dessous selon votre installation
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Exemple pour Linux
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Exemple pour Windows

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_crawler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    A utility class for extracting text from PDF files.

    This class handles the conversion of PDFs to images for OCR processing,
    extracts text using both OCR and direct extraction methods, and processes
    the text with GPT-4 for structuring and formatting. It supports
    multithreading for efficient processing of large documents.

    Attributes:
        input_dir (Path): Directory containing input files (PDFs).
        output_dir (Path): Directory where extracted and processed content is saved.
        api_keys (Queue): Queue of OpenAI API keys for handling rate limits.
        max_workers (int): Maximum number of worker threads for concurrent processing.
        initial_dpi (int): Initial DPI setting for PDF to image conversion.
        retry_dpi (int): DPI setting for retrying failed conversions.
        logger (logging.Logger): Logger for recording actions and errors.
    """

    def __init__(self, input_dir, output_dir, api_keys_file, max_workers=10, initial_dpi=300, retry_dpi=200, logger=None):
        """
        Initialize the PDFExtractor with necessary configurations.

        Args:
            input_dir (str or Path): Directory containing input PDF files.
            output_dir (str or Path): Directory to save extracted content.
            api_keys_file (str): Path to the file containing OpenAI API keys.
            max_workers (int, optional): Number of threads for parallel processing. Defaults to 10.
            initial_dpi (int, optional): DPI for initial PDF to image conversion. Defaults to 300.
            retry_dpi (int, optional): DPI for retry attempts if initial conversion fails. Defaults to 200.
            logger (logging.Logger, optional): Logger for logging information and errors. Defaults to None.
        """
        # Configure paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load OpenAI API keys from file
        self.api_keys = Queue()
        try:
            with open(api_keys_file, 'r') as f:
                for line in f:
                    key = line.strip()
                    if key:
                        self.api_keys.put(key)
            if self.api_keys.empty():
                raise ValueError("No API keys loaded.")
        except Exception as e:
            if logger:
                logger.error(f"🚫 Error loading API keys: {str(e)}")
            else:
                print(f"🚫 Error loading API keys: {str(e)}")
            raise

        self.max_workers = max_workers
        self.initial_dpi = initial_dpi
        self.retry_dpi = retry_dpi
        self.logger = logger or logging.getLogger(__name__)

    def preprocess_image(self, image):
        """
        Preprocess an image to enhance OCR accuracy.

        The preprocessing steps include converting to grayscale, denoising, applying CLAHE,
        and adaptive thresholding. These steps improve the quality of text recognition.

        Args:
            image (PIL.Image.Image or np.ndarray): The image to preprocess.

        Returns:
            np.ndarray: The preprocessed binary image ready for OCR.

        Raises:
            ValueError: If the image is empty or corrupted, or if preprocessing fails.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        if image is None:
            raise ValueError("⚠ Empty or corrupted image.")

        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            denoised = cv2.fastNlMeansDenoising(gray, h=30)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            return binary
        except Exception as e:
            raise ValueError(f"⚠ Error during image preprocessing: {str(e)}")

    def convert_pdf_to_images(self, pdf_path, dpi):
        """
        Convert a PDF file into a list of images at a specified DPI.

        Args:
            pdf_path (Path): Path to the PDF file to convert.
            dpi (int): Dots per inch for the conversion resolution.

        Returns:
            list or None: List of PIL.Image.Image objects if successful, None otherwise.
        """
        try:
            self.logger.info(f"📄 Converting {pdf_path.name} to images with DPI={dpi}")
            images = convert_from_path(pdf_path, dpi=dpi, fmt='jpeg', thread_count=1)
            self.logger.info(f"✅ Successfully converted {pdf_path.name} with DPI={dpi}")
            return images
        except Exception as e:
            self.logger.error(f"❌ Error converting {pdf_path.name} to images with DPI={dpi}: {str(e)}")
            return None

    def extract_text_with_ocr(self, pdf_path):
        """
        Extract text from a PDF using OCR with retry mechanisms for failed conversions.

        The method first attempts to convert the PDF to images at the initial DPI setting.
        If unsuccessful, it retries with a lower DPI. Subsequently, it performs OCR on each image,
        with fallback OCR configurations if initial attempts yield insufficient text.

        Args:
            pdf_path (Path): Path to the PDF file.

        Returns:
            list or None: List of extracted text strings per page if successful, None otherwise.
        """
        # First attempt with the initial DPI
        images = self.convert_pdf_to_images(pdf_path, self.initial_dpi)
        if images is None:
            # Second attempt with a reduced DPI
            images = self.convert_pdf_to_images(pdf_path, self.retry_dpi)
            if images is None:
                self.logger.error(f"❌ Failed to convert {pdf_path.name} to images with all attempted DPIs.")
                return None

        ocr_texts = []
        for i, image in enumerate(images, 1):
            self.logger.info(f"🔍 Performing OCR on page {i}/{len(images)} of {pdf_path.name}")
            try:
                processed_img = self.preprocess_image(image)
            except Exception as e:
                self.logger.error(f"⚠️ Image preprocessing failed for page {i} of {pdf_path.name}: {str(e)}")
                ocr_texts.append("")
                continue

            try:
                text = pytesseract.image_to_string(
                    processed_img,
                    lang='fra+eng',  # Ajustez les langues selon vos besoins
                    config='--psm 1'
                )
                if len(text.strip()) < 100:
                    self.logger.info(f"🔄 Insufficient OCR output for page {i} of {pdf_path.name}, retrying with different config")
                    text = pytesseract.image_to_string(
                        processed_img,
                        lang='fra+eng',
                        config='--psm 3 --oem 1'
                    )
                ocr_texts.append(text)
            except Exception as e:
                self.logger.error(f"❌ OCR failed for page {i} of {pdf_path.name}: {str(e)}")
                ocr_texts.append("")

        return ocr_texts

    def extract_text_with_pypdf_per_page(self, pdf_path, page_num):
        """
        Extract text from a specific page of a PDF using PyPDF.

        Args:
            pdf_path (Path): Path to the PDF file.
            page_num (int): The page number from which to extract text.

        Returns:
            str: Extracted text if successful, empty string otherwise.
        """
        try:
            with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                if page_num < 1 or page_num > len(reader.pages):
                    self.logger.error(f"⚠️ Invalid page number: {page_num} in {pdf_path.name}")
                    return ''
                page = reader.pages[page_num - 1]
                text = page.extract_text() or ''
                self.logger.info(f"📝 Extracted PyPDF text from page {page_num} of {pdf_path.name}: {len(text)} characters")
                return text
        except Exception as e:
            self.logger.error(f"❌ PyPDF error on page {page_num} of {pdf_path.name}: {str(e)}")
            return ''

    def get_api_key(self):
        """
        Retrieve an API key from the queue.

        This method fetches an API key from the `api_keys` queue to handle
        API rate limiting by cycling through available keys.

        Returns:
            str or None: An API key if available, None otherwise.
        """
        try:
            api_key = self.api_keys.get(timeout=10)
            return api_key
        except Empty:
            self.logger.error("🚫 No API keys available.")
            return None

    def process_with_gpt(self, content):
        """
        Process the given text content using GPT-4 to structure it into Markdown.

        The method sends the content to the GPT-4 model with a system prompt guiding
        the formatting and structuring rules. It handles API responses and rate limiting.

        Args:
            content (str): The raw text content to be processed.

        Returns:
            str or None: The structured Markdown content if successful, None otherwise.
        """
        system_prompt = {
            "role": "system",
            "content": (
                "You are a document analysis expert. Your task is to: "
                "1. Extract and structure key information from the provided text following these rules:\n"
                "   - Create a clear hierarchy with titles (# ## ###)\n"
                "   - Separate sections with line breaks\n"
                "   - Ensure consistency in presentation\n\n"
                "2. For tables:\n"
                "   - Convert each row into list items\n"
                "   - Use the format '- **[Column Name]:** [Value]'\n"
                "   - Group related items with indentation\n"
                "   - Add '---' separators between groups\n\n"
                "3. Apply the following formatting:\n"
                "   - Use italics (*) for important terms\n"
                "   - Use bold (**) for column headers\n"
                "   - Create bullet lists (-) for enumerations\n"
                "   - Use blockquotes (>) for important notes\n\n"
                "4. Clean and improve the text:\n"
                "   - Correct OCR typos\n"
                "   - Unify punctuation\n"
                "   - Remove unwanted characters\n"
                "   - Check alignment and spacing\n\n"
                "Example transformation:\n"
                "Table: 'Product | Price | Stock\n"
                "         Apples | 2.50 | 100\n"
                "         Pears | 3.00 | 85'\n\n"
                "Becomes:\n"
                "### Product List\n\n"
                "- **Product:** Apples\n"
                "  - **Price:** 2.50€\n"
                "  - **Stock:** 100 units\n\n"
                "---\n\n"
                "- **Product:** Pears\n"
                "  - **Price:** 3.00€\n"
                "  - **Stock:** 85 units"
            )
        }

        api_key = self.get_api_key()
        if not api_key:
            return None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "gpt-4",
            "messages": [
                system_prompt,
                {"role": "user", "content": content}
            ],
            "temperature": 0,
            "max_tokens": 16000,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        try:
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            processed_content = response.json()['choices'][0]['message']['content']
            time.sleep(1)  # Pause to respect rate limits
            return processed_content
        except Exception as e:
            self.logger.error(f"❌ GPT API error: {str(e)}")
            return None
        finally:
            # Return the API key back to the queue
            if api_key:
                self.api_keys.put(api_key)

    def split_content(self, content, max_length=4000):
        """
        Split the content into smaller chunks if it exceeds the maximum length.

        The method ensures that each chunk is within the specified `max_length` by splitting
        based on paragraph breaks and maintaining an overlap for context continuity.

        Args:
            content (str): The text content to split.
            max_length (int, optional): Maximum length of each chunk. Defaults to 4000.

        Returns:
            list: A list of text chunks.
        """
        try:
            paragraphs = content.split('\n\n')
            parts = []
            current_part = ""
            for para in paragraphs:
                if len(current_part) + len(para) + 2 > max_length:
                    if current_part:
                        parts.append(current_part.strip())
                    current_part = para + '\n\n'
                else:
                    current_part += para + '\n\n'
            if current_part.strip():
                parts.append(current_part.strip())
            return parts
        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            return [content]  # Return the full text in case of error

    def process_single_part(self, document_name, page_num, part_num, content):
        """
        Process a single part of the content with GPT-4.

        This method structures the content into Markdown and saves it to an output file.

        Args:
            document_name (str): Name of the document being processed.
            page_num (int): Page number of the document.
            part_num (int): Part number within the page.
            content (str): The text content to process.
        """
        self.logger.info(f"📝 Processing Document: {document_name}, Page: {page_num}, Part: {part_num}")
        processed_content = self.process_with_gpt(content)

        if processed_content:
            output_file_name = self.output_dir / f"{document_name}_page_{page_num}_part_{part_num}.txt"
            try:
                with open(output_file_name, 'a', encoding='utf-8') as f:
                    f.write(f"📄 Document ID: {document_name}\n\n{processed_content}\n\n")
                self.logger.info(f"✅ File created: {output_file_name.name}")
            except Exception as e:
                self.logger.error(f"❌ Error saving Document: {document_name}, Page: {page_num}, Part: {part_num}: {str(e)}")

    def process_pdf(self, pdf_path):
        """
        Process an individual PDF file by extracting text and structuring it.

        The method handles both OCR-based and direct text extraction, splits the content
        into manageable chunks, and processes each chunk with GPT-4.

        Args:
            pdf_path (Path): Path to the PDF file to process.

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        document_name = pdf_path.stem
        self.logger.info(f"📂 Starting processing of {pdf_path.name}")

        # Extract text using OCR
        ocr_texts = self.extract_text_with_ocr(pdf_path)
        if ocr_texts is None:
            self.logger.error(f"❌ OCR extraction failed for {pdf_path.name}")
            return False

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for page_num, ocr_text in enumerate(ocr_texts, 1):
                self.logger.info(f"🔄 Preparing page {page_num}/{len(ocr_texts)} of {pdf_path.name}")

                if ocr_text and len(ocr_text.strip()) >= 100:
                    page_text = ocr_text
                    self.logger.info(f"✅ OCR succeeded for page {page_num} of {pdf_path.name}")
                else:
                    self.logger.info(f"🔄 Insufficient OCR for page {page_num} of {pdf_path.name}, using PyPDF")
                    pypdf_text = self.extract_text_with_pypdf_per_page(pdf_path, page_num)
                    page_text = pypdf_text

                if not page_text.strip():
                    self.logger.warning(f"⚠️ No text extracted for page {page_num} of {pdf_path.name}")
                    continue  # Skip to next page if no text is extracted

                # Split text if too long
                parts = self.split_content(page_text, max_length=4000)  # Adjust max_length if necessary

                for idx, part in enumerate(parts, 1):
                    futures.append(executor.submit(
                        self.process_single_part, document_name, page_num, idx, part
                    ))

            # Wait for all tasks to complete
            for future in as_completed(futures):
                pass  # Logging is handled within individual tasks

        self.logger.info(f"✅ Completed processing of {pdf_path.name}")
        return True

    def process_all_pdfs(self):
        """
        Process all PDF files in the input directory.

        The method iterates through each PDF file, processing them concurrently
        using a thread pool for efficiency. Progress is displayed using a tqdm progress bar.
        """
        pdf_files = list(self.input_dir.glob('*.pdf'))
        total_files = len(pdf_files)

        if total_files == 0:
            self.logger.warning(f"Aucun fichier PDF trouvé dans le dossier : {self.input_dir}")
            return

        self.logger.info(f"📢 Starting processing of {total_files} PDF files in '{self.input_dir}'")

        with tqdm(total=total_files, desc="Traitement des PDFs", unit="pdf") as pbar:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_pdf = {executor.submit(self.process_pdf, pdf_path): pdf_path for pdf_path in pdf_files}
                for future in as_completed(future_to_pdf):
                    pdf_path = future_to_pdf[future]
                    try:
                        success = future.result()
                        if success:
                            self.logger.info(f"✅ Successfully processed {pdf_path.name}")
                        else:
                            self.logger.warning(f"⚠️ Failed to process {pdf_path.name}")
                    except Exception as e:
                        self.logger.error(f"❌ Exception occurred while processing {pdf_path.name}: {str(e)}")
                    finally:
                        pbar.update(1)

        self.logger.info(f"🎉 Completed. Processed {total_files} PDF files.")

def main():
    """
    Fonction principale pour traiter tous les fichiers PDF dans le dossier spécifié.
    """
    # Définir les chemins
    PDF_FOLDER = Path("path/to/your/pdf_folder")       # Remplacez par le chemin de votre dossier de PDF
    OUTPUT_FOLDER = Path("path/to/output_folder")     # Remplacez par le chemin où vous souhaitez sauvegarder les résultats
    API_KEYS_FILE = "api_keys.txt"                    # Fichier contenant vos clés API OpenAI, une par ligne

    # Vérifier l'existence des dossiers
    if not PDF_FOLDER.exists():
        logger.error(f"Le dossier PDF spécifié n'existe pas : {PDF_FOLDER}")
        return

    if not Path(API_KEYS_FILE).exists():
        logger.error(f"Le fichier de clés API spécifié n'existe pas : {API_KEYS_FILE}")
        return

    # Initialiser le PDFExtractor
    extractor = PDFExtractor(
        input_dir=PDF_FOLDER,
        output_dir=OUTPUT_FOLDER,
        api_keys_file=API_KEYS_FILE,
        max_workers=4,           # Ajustez selon les capacités de votre machine
        initial_dpi=300,
        retry_dpi=200,
        logger=logger
    )

    # Démarrer le traitement
    extractor.process_all_pdfs()

if __name__ == "__main__":
    main()
