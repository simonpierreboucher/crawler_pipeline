import os
import re
import sys
import time
import hashlib
import logging
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from colorama import init, Fore, Style
from tqdm import tqdm
import html2text

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.traceback import install

# Initialize Rich modules for improved tracebacks
install()

# Initialize Colorama for colored terminal text
init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """
    A custom logging formatter that adds colors and symbols to enhance log readability.
    """

    # Define colors for different log levels
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    # Define symbols for different log levels
    SYMBOLS = {
        'INFO': '✔',
        'WARNING': '⚠',
        'ERROR': '✘',
        'CRITICAL': '✘'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, Fore.WHITE)
        symbol = self.SYMBOLS.get(record.levelname, '')
        # Initial format with symbol and timestamp, followed by '#'
        header = f"{color}{symbol} {self.formatTime(record, self.datefmt)}#"
        # Log level, followed by '#'
        level = f"- {record.levelname}#"
        # Log message, followed by '#'
        message = f"- {record.getMessage()}#"
        return f"{header}\n{level}\n{message}"


class MovingIndicator(threading.Thread):
    """
    A thread that displays a moving indicator in the terminal to signify ongoing processes.
    """

    def __init__(self, delay=0.1, length=20):
        super().__init__()
        self.delay = delay
        self.length = length
        self.running = False
        self.position = self.length - 1  # Start from the right
        self.direction = -1  # Move left

    def run(self):
        self.running = True
        while self.running:
            # Create the representation of the indicator
            line = [' '] * self.length
            if 0 <= self.position < self.length:
                line[self.position] = '#'
            indicator = ''.join(line) + '##'  # Add '##' at the end for dynamism
            # Display the indicator
            sys.stdout.write(f"\r{indicator}")
            sys.stdout.flush()
            # Update the position
            self.position += self.direction
            if self.position == 0 or self.position == self.length - 1:
                self.direction *= -1  # Change direction
            time.sleep(self.delay)

    def stop(self):
        self.running = False
        self.join()
        # Clean the indicator line
        sys.stdout.write('\r' + ' ' * (self.length + 2) + '\r')  # +2 for the '##'
        sys.stdout.flush()


class WebCrawler:
    """
    A comprehensive web crawler for extracting content and downloadable files from websites.
    """

    def __init__(self, start_url, max_depth=2, use_playwright=False, excluded_paths=None,
                 download_extensions=None, language_pattern=None, base_dir=None):
        self.start_url = start_url
        self.max_depth = max_depth
        self.use_playwright = use_playwright
        self.visited_pages = set()
        self.downloaded_files = set()
        self.domain = urlparse(start_url).netloc
        self.excluded_paths = excluded_paths or ['selecteur-de-produits']
        self.download_extensions = download_extensions or {
            'PDF': ['.pdf'],
            'Image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
            'Doc': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
        }
        self.language_pattern = language_pattern
        self.base_dir = base_dir or f"crawler_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.create_directories()
        self.setup_logging()
        self.stats = defaultdict(int)
        self.all_downloadable_exts = set(ext for exts in self.downloadable_extensions.values() for ext in exts)
        self.content_type_mapping = {
            'PDF': {
                'application/pdf': '.pdf'
            },
            'Image': {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/svg+xml': '.svg',
            },
            'Doc': {
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'application/vnd.ms-excel': '.xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
                'application/vnd.ms-powerpoint': '.ppt',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            }
        }
        self.session = self.setup_session()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.body_width = 0
        self.html_converter.ignore_images = True
        self.html_converter.single_line_break = False

        if self.use_playwright:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()

        self.indicator = MovingIndicator(length=20)

    def setup_session(self):
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = False  # Disable SSL verification if necessary
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                          'AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/91.0.4472.124 Safari/537.36'
        })
        return session

    def create_directories(self):
        directories = ['content', 'PDF', 'Image', 'Doc', 'logs']
        for dir_name in directories:
            path = os.path.join(self.base_dir, dir_name)
            os.makedirs(path, exist_ok=True)

    def setup_logging(self):
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = ColoredFormatter(log_format)

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # File handler for logging to a file without colors
        file_handler = logging.FileHandler(os.path.join(self.base_dir, 'logs', 'crawler.log'), encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

        # Console handler for logging to the terminal with colors
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Log the start of the crawler
        logging.info(f"Crawler started with language pattern: {self.language_pattern}")

    def should_exclude(self, url):
        for excluded in self.excluded_paths:
            if excluded in url:
                return True
        return False

    def is_same_language(self, url):
        if not self.language_pattern:
            return True
        return self.language_pattern.search(url) is not None

    def is_downloadable_file(self, url):
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        pattern = re.compile(r'\.(' + '|'.join([ext.strip('.') for exts in self.downloadable_extensions.values() for ext in exts]) + r')(\.[a-z0-9]+)?$', re.IGNORECASE)
        return bool(pattern.search(path))

    def get_file_type_and_extension(self, url, response):
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()

        # First attempt: deduce file type based on the URL
        for file_type, extensions in self.downloadable_extensions.items():
            for ext in extensions:
                pattern = re.compile(re.escape(ext) + r'(\.[a-z0-9]+)?$', re.IGNORECASE)
                if pattern.search(path):
                    return file_type, self.content_type_mapping[file_type].get(response.headers.get('Content-Type', '').lower(), ext)

        # Second attempt: deduce file type based on the Content-Type
        content_type = response.headers.get('Content-Type', '').lower()
        for file_type, mapping in self.content_type_mapping.items():
            if content_type in mapping:
                return file_type, mapping[content_type]

        # If no type is determined, return None
        return None, None

    def sanitize_filename(self, url, file_type, extension, page_number=None):
        # Create a short hash of the URL
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

        # Extract the last segment of the URL
        filename = url.split('/')[-1]
        if not filename:
            filename = 'index'

        # Clean the filename by replacing unsafe characters
        filename = re.sub(r'[^\w\-_.]', '_', filename)

        # Remove existing extensions
        name, _ = os.path.splitext(filename)

        # Define the extension based on file type
        if not extension:
            extension = '.txt'  # Default extension if not determined

        if page_number is not None:
            sanitized = f"{name}_page_{page_number:03d}_{url_hash}{extension}"
        else:
            sanitized = f"{name}_{url_hash}{extension}"

        logging.debug(f"Sanitized filename: {sanitized}")
        return sanitized

    def download_file(self, url, file_type):
        try:
            logging.info(f"Attempting to download {file_type} from: {url}")

            # Determine the file type and extension
            response = self.session.head(url, allow_redirects=True, timeout=10)
            file_type_detected, extension = self.get_file_type_and_extension(url, response)
            if not file_type_detected:
                logging.warning(f"⚠ Unable to determine file type for: {url}")
                return False

            # Properly rename the file
            filename = self.sanitize_filename(url, file_type_detected, extension)
            save_path = os.path.join(self.base_dir, file_type_detected, filename)

            # Check if the file already exists
            if os.path.exists(save_path):
                logging.info(f"📂 File already downloaded, skipping: {filename}")
                return False

            # Download the file with a progress bar
            response = self.session.get(url, stream=True, timeout=20)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kibibyte
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"⏬ Downloading {filename}", leave=False)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            progress_bar.close()

            if total_size != 0 and progress_bar.n != total_size:
                logging.warning(f"⚠ Incomplete download for {url}")
                return False

            self.stats[f'{file_type_detected}_downloaded'] += 1
            self.downloaded_files.add(url)
            logging.info(f"✅ Successfully downloaded {file_type_detected}: {filename}")
            return True

        except Exception as e:
            logging.error(f"✘ Error downloading {url}: {str(e)}")
            return False

    def convert_links_to_absolute(self, soup, base_url):
        # Include <embed>, <iframe>, and <object> tags in addition to <a> and <link>
        for tag in soup.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            href = tag.get('href') or tag.get('src')
            if href:
                absolute_url = urljoin(base_url, href)
                if tag.name in ['embed', 'iframe', 'object']:
                    tag['src'] = absolute_url
                else:
                    tag['href'] = absolute_url
        return soup

    def clean_text(self, text):
        if not text:
            return ""

        # Remove unwanted special characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)

        # Normalize spaces (replace multiple spaces/tabs with a single space)
        text = re.sub(r'[ \t]+', ' ', text)

        # Normalize newlines (replace multiple newlines with two newlines)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()

    def fetch_page_content(self, url):
        if self.use_playwright:
            try:
                logging.info(f"🔍 Fetching with Playwright: {url}")
                self.page.goto(url, timeout=20000)
                time.sleep(2)  # Wait for the page to fully load
                content = self.page.content()
                return content
            except Exception as e:
                logging.error(f"✘ Playwright failed to fetch {url}: {str(e)}")
                return None
        else:
            try:
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    logging.info(f"✅ Successfully fetched content: {url}")
                    return response.text
                else:
                    logging.warning(f"⚠ Failed to fetch {url} with status code {response.status_code}")
                    return None
            except Exception as e:
                logging.error(f"✘ Requests failed to fetch {url}: {str(e)}")
                return None

    def extract_content(self, url):
        logging.info(f"📄 Extracting content from: {url}")

        try:
            if self.is_downloadable_file(url):
                logging.info(f"🔗 Skipping content extraction for downloadable file: {url}")
                return

            page_content = self.fetch_page_content(url)
            if page_content is None:
                logging.warning(f"⚠ Unable to retrieve content for: {url}")
                return

            soup = BeautifulSoup(page_content, 'html.parser')

            # Remove unwanted elements
            for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside', 'iframe']):
                element.decompose()

            # Extract main content
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_='content') or
                soup.find('div', id='content')
            )

            if main_content:
                # Convert relative links to absolute
                main_content = self.convert_links_to_absolute(main_content, url)

                # Convert main content to Markdown
                markdown_content = self.html_converter.handle(str(main_content))

                # Build the final content with title
                content_parts = []

                # Add the title if available
                title = soup.find('h1')
                if title:
                    content_parts.append(f"# {title.get_text().strip()}")

                # Add the source URL
                content_parts.append(f"**Source:** {url}")

                # Add the main Markdown content
                content_parts.append(markdown_content)

                # Clean the content
                content = self.clean_text('\n\n'.join(content_parts))

                if content:
                    # Create filename
                    filename = self.sanitize_filename(url, 'Doc', '.txt')  # Use 'Doc' with '.txt' extension for pages
                    save_path = os.path.join(self.base_dir, 'content', filename)
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(content)

                    self.stats['pages_processed'] += 1
                    logging.info(f"✅ Content successfully saved to: {filename}")
                else:
                    logging.warning(f"⚠ No significant content found for: {url}")

                # Process downloadable files found on the page
                downloadable_tags = main_content.find_all(['a', 'embed', 'iframe', 'object'], href=True)

                if downloadable_tags:
                    logging.info(f"🔄 Detected {len(downloadable_tags)} downloadable files on the page.")

                for tag in downloadable_tags:
                    href = tag.get('href') or tag.get('src')
                    if href:
                        file_url = urljoin(url, href)
                        if self.is_downloadable_file(file_url) and file_url not in self.downloaded_files:
                            # Use HEAD to get Content-Type without downloading the file
                            try:
                                response_head = self.session.head(file_url, allow_redirects=True, timeout=10)
                                file_type_detected, _ = self.get_file_type_and_extension(file_url, response_head)
                            except:
                                # Fallback to GET if HEAD fails
                                response_head = self.session.get(file_url, allow_redirects=True, timeout=10)
                                file_type_detected, _ = self.get_file_type_and_extension(file_url, response_head)

                            if file_type_detected:
                                # Properly rename the file
                                filename = self.sanitize_filename(file_url, file_type_detected,
                                                                  self.content_type_mapping[file_type_detected]
                                                                  .get(response_head.headers.get('Content-Type', '').lower(), ''))
                                save_path = os.path.join(self.base_dir, file_type_detected, filename)

                                if os.path.exists(save_path):
                                    logging.info(f"📂 File already downloaded, skipping: {filename}")
                                    continue

                                self.download_file(file_url, file_type_detected)

            else:
                logging.warning(f"⚠ No main content found for: {url}")

        except Exception as e:
            logging.error(f"✘ Error processing {url}: {str(e)}")

    def extract_urls(self, start_url):
        queue = deque()
        queue.append((start_url, 0))
        self.visited_pages.add(start_url)

        pbar = tqdm(total=1, desc="🔍 Extracting URLs", unit="page", ncols=100)

        while queue:
            current_url, depth = queue.popleft()

            if depth > self.max_depth:
                pbar.update(1)
                continue

            # Check if the URL should be excluded
            if self.should_exclude(current_url):
                logging.info(f"🚫 URL excluded due to excluded segment: {current_url}")
                pbar.update(1)
                continue

            logging.info(f"🔎 Extracting URLs from: {current_url} (depth: {depth})")

            try:
                if self.is_downloadable_file(current_url):
                    # Use HEAD to get Content-Type without downloading the file
                    try:
                        response_head = self.session.head(current_url, allow_redirects=True, timeout=10)
                        file_type_detected, _ = self.get_file_type_and_extension(current_url, response_head)
                    except:
                        # Fallback to GET if HEAD fails
                        response_head = self.session.get(current_url, allow_redirects=True, timeout=10)
                        file_type_detected, _ = self.get_file_type_and_extension(current_url, response_head)

                    if file_type_detected:
                        # Rename the file to check existence
                        filename = self.sanitize_filename(current_url, file_type_detected,
                                                          self.content_type_mapping[file_type_detected]
                                                          .get(response_head.headers.get('Content-Type', '').lower(),
                                                               ''))
                        save_path = os.path.join(self.base_dir, file_type_detected, filename)

                        if os.path.exists(save_path):
                            logging.info(f"📂 File already downloaded, skipping: {filename}")
                            pbar.update(1)
                            continue

                        self.download_file(current_url, file_type_detected)
                        self.downloaded_files.add(current_url)
                    pbar.update(1)
                    continue

                # Fetch page content (with Playwright or Requests)
                page_content = self.fetch_page_content(current_url)
                if page_content is None:
                    logging.warning(f"⚠ Unable to retrieve content for: {current_url}")
                    pbar.update(1)
                    continue

                soup = BeautifulSoup(page_content, 'html.parser')

                # Search for downloadable files and links
                for tag in soup.find_all(['a', 'link', 'embed', 'iframe', 'object'], href=True):
                    href = tag.get('href') or tag.get('src')
                    if href:
                        absolute_url = urljoin(current_url, href)
                        parsed_url = urlparse(absolute_url)

                        # Check if it's a downloadable file
                        if self.is_downloadable_file(absolute_url):
                            # Use HEAD to get Content-Type without downloading the file
                            try:
                                response_head = self.session.head(absolute_url, allow_redirects=True, timeout=10)
                                file_type_detected, _ = self.get_file_type_and_extension(absolute_url, response_head)
                            except:
                                # Fallback to GET if HEAD fails
                                response_head = self.session.get(absolute_url, allow_redirects=True, timeout=10)
                                file_type_detected, _ = self.get_file_type_and_extension(absolute_url, response_head)

                            if file_type_detected:
                                # Rename the file to check existence
                                filename = self.sanitize_filename(absolute_url, file_type_detected,
                                                                  self.content_type_mapping[file_type_detected]
                                                                  .get(response_head.headers.get('Content-Type', '').lower(),
                                                                       ''))
                                save_path = os.path.join(self.base_dir, file_type_detected, filename)

                                if os.path.exists(save_path):
                                    logging.info(f"📂 File already downloaded, skipping: {filename}")
                                    continue

                                self.download_file(absolute_url, file_type_detected)
                                self.downloaded_files.add(absolute_url)
                            continue

                        # Check for internal links
                        if (self.domain in parsed_url.netloc and
                            self.is_same_language(absolute_url) and
                            absolute_url not in self.visited_pages and
                            not absolute_url.endswith(('#', 'javascript:void(0)', 'javascript:;')) and
                            not self.should_exclude(absolute_url)):

                            # Add to queue with incremented depth
                            queue.append((absolute_url, depth + 1))
                            self.visited_pages.add(absolute_url)
                            pbar.total += 1  # Increase progress bar
                            pbar.refresh()

            except Exception as e:
                logging.error(f"✘ Error crawling {current_url}: {str(e)}")

            pbar.update(1)

        pbar.close()
        logging.info("🔍 URL extraction completed.")

    def crawl(self):
        start_time = time.time()
        logging.info(f"🚀 Starting crawl of {self.start_url}")
        logging.info(f"🌐 Language pattern: {self.language_pattern}")
        logging.info(f"📏 Maximum depth: {self.max_depth}")

        # Load previously downloaded files
        self.load_downloaded_files()

        # Start the moving indicator
        self.indicator.start()

        try:
            # Phase 1: URL Extraction
            logging.info("🔍 Phase 1: Starting URL extraction")
            self.extract_urls(self.start_url)

            # Phase 2: Content Extraction
            logging.info("📄 Phase 2: Starting content extraction")
            visited_without_files = [url for url in self.visited_pages if not self.is_downloadable_file(url)]

            pbar_content = tqdm(total=len(visited_without_files), desc="📄 Extracting Content", unit="page", ncols=100)
            for i, url in enumerate(visited_without_files, 1):
                logging.info(f"📝 Processing URL {i}/{len(visited_without_files)}: {url}")
                self.extract_content(url)
                pbar_content.update(1)
            pbar_content.close()
            logging.info("📄 Content extraction completed.")

            # Generate the final report
            end_time = time.time()
            self.generate_report(end_time - start_time)

        except Exception as e:
            logging.error(f"⚠ Critical error during crawling: {str(e)}")
            self.generate_report(time.time() - start_time, error=str(e))

        finally:
            # Stop the moving indicator
            self.indicator.stop()

            # Save downloaded files
            self.save_downloaded_files()

            # Close Playwright if used
            if self.use_playwright:
                self.page.close()
                self.browser.close()
                self.playwright.stop()

    def load_downloaded_files(self):
        downloaded_files_path = os.path.join(self.base_dir, 'logs', 'downloaded_files.txt')
        if os.path.exists(downloaded_files_path):
            with open(downloaded_files_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.downloaded_files.add(line.strip())
            logging.info(f"📥 Loaded {len(self.downloaded_files)} downloaded files from the tracking file.")
        else:
            logging.info("🆕 No download tracking file found, starting fresh.")

    def save_downloaded_files(self):
        downloaded_files_path = os.path.join(self.base_dir, 'logs', 'downloaded_files.txt')
        try:
            with open(downloaded_files_path, 'w', encoding='utf-8') as f:
                for url in sorted(self.downloaded_files):
                    f.write(url + '\n')
            logging.info(f"💾 Saved {len(self.downloaded_files)} downloaded files to the tracking file.")
        except Exception as e:
            logging.error(f"✘ Error saving downloaded files tracking: {str(e)}")

    def generate_report(self, duration, error=None):
        report_sections = []

        # Report header
        report_sections.append(f"""
Crawler Report
==============

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration
------------
Start URL: {self.start_url}
Language Pattern: {self.language_pattern}
Max Depth: {self.max_depth}
Duration: {duration:.2f} seconds

Statistics
---------
Total URLs found: {len(self.visited_pages)}
Pages processed: {self.stats['pages_processed']}
Files downloaded:
- PDFs: {self.stats['PDF_downloaded']}
- Images: {self.stats['Image_downloaded']}
- Documents: {self.stats['Doc_downloaded']}
""")

        # Error section if present
        if error:
            report_sections.append(f"""
Errors
------
Critical Error: {error}

""")

        # List of processed URLs
        report_sections.append("""
Processed URLs
-------------
""")
        for url in sorted(self.visited_pages):
            report_sections.append(url)

        # List of generated files
        report_sections.append("""
Generated Files
--------------
""")
        for directory in ['content', 'PDF', 'Image', 'Doc']:
            dir_path = os.path.join(self.base_dir, directory)
            if os.path.exists(dir_path):
                files = os.listdir(dir_path)
                report_sections.append(f"\n{directory} Files ({len(files)}):")
                for file in sorted(files):
                    report_sections.append(f"- {file}")

        # Save the report
        report_content = '\n'.join(report_sections)
        report_path = os.path.join(self.base_dir, 'crawler_report.txt')

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"📄 Report successfully generated: {report_path}")
        except Exception as e:
            logging.error(f"✘ Error generating report: {str(e)}")

        # Create a summary file
        summary = f"""
Crawling Summary
---------------  
Start URL: {self.start_url}  
Total URLs: {len(self.visited_pages)}  
Pages Processed: {self.stats['pages_processed']}  
Total Files Downloaded: {sum(self.stats[k] for k in ['PDF_downloaded', 'Image_downloaded', 'Doc_downloaded'])}  
Duration: {duration:.2f} seconds  
Status: {'⚠ Completed with errors' if error else '✅ Completed successfully'}  
"""
        try:
            with open(os.path.join(self.base_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
                f.write(summary)
            logging.info(f"📄 Summary successfully generated: {os.path.join(self.base_dir, 'summary.txt')}")
        except Exception as e:
            logging.error(f"✘ Error generating summary: {str(e)}")
