import os
import re
import json
import logging
import threading
import requests
import numpy as np
from pathlib import Path
from queue import Queue, Empty
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

class EmbeddingProcessor:
    """
    A class for processing text embeddings using OpenAI's API.

    This processor handles the chunking of text, contextualization using GPT-4,
    and embedding generation. It manages API rate limits by cycling through
    multiple API keys and supports concurrent processing for efficiency.
    """

    def __init__(self, input_dir, output_dir, openai_api_keys, verbose=False, logger=None):
        """
        Initialize the EmbeddingProcessor with necessary configurations.

        Args:
            input_dir (str or Path): Directory containing input text files.
            output_dir (str or Path): Directory where embeddings and metadata are saved.
            openai_api_keys (list): List of OpenAI API keys for cycling.
            verbose (bool, optional): Flag to enable verbose logging. Defaults to False.
            logger (logging.Logger, optional): Logger for logging information and errors. Defaults to None.
        """
        # Configure paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize global lists for all files
        self.all_embeddings = []
        self.all_metadata = []

        # Configure OpenAI API
        self.openai_api_keys = openai_api_keys
        self.headers_cycle = cycle([
            {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json"
            } for key in self.openai_api_keys
        ])
        self.lock = threading.Lock()

        # Configure logging
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose

    def chunk_text(self, text, chunk_size=1200, overlap_size=100):
        """
        Split text into smaller chunks with overlapping regions.

        Args:
            text (str): The text to be split.
            chunk_size (int, optional): Maximum number of tokens per chunk. Defaults to 1200.
            overlap_size (int, optional): Number of overlapping tokens between chunks. Defaults to 100.

        Returns:
            list: List of text chunks.
        """
        try:
            tokens = text.split()

            # If text is shorter than chunk size, process as a single chunk
            if len(tokens) <= chunk_size:
                return [text]

            chunks = []
            for i in range(0, len(tokens), chunk_size - overlap_size):
                chunk = ' '.join(tokens[i:i + chunk_size])
                chunks.append(chunk)

            # Ensure the last chunk isn't too small
            if len(chunks) > 1 and len(tokens[-(chunk_size - overlap_size):]) < chunk_size // 2:
                # Merge the last chunk with the previous one if it's too small
                last_chunk = ' '.join(tokens[-chunk_size:])
                chunks[-1] = last_chunk

            return chunks

        except Exception as e:
            self.logger.error(f"Error splitting text: {str(e)}")
            return [text]  # Return full text in case of error

    def get_embedding(self, text, headers, document_name, page_num, chunk_id):
        """
        Obtain an embedding vector for a given text using OpenAI's Embedding API.

        Args:
            text (str): The text to embed.
            headers (dict): HTTP headers containing the API authorization.
            document_name (str): Name of the document being processed.
            page_num (int): Page number within the document.
            chunk_id (int): Identifier for the chunk within the page.

        Returns:
            list or None: The embedding vector if successful, None otherwise.
        """
        try:
            payload = {
                "input": text,
                "model": "text-embedding-ada-002",  # Use the appropriate embedding model
                "encoding_format": "float"
            }

            if self.verbose:
                self.logger.info(f"Calling Embedding API for {document_name} page {page_num} chunk {chunk_id}")

            self.logger.info(f"🔗 Calling Embedding API for {document_name} page {page_num} chunk {chunk_id}")

            response = requests.post(
                'https://api.openai.com/v1/embeddings',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()['data'][0]['embedding']

        except Exception as e:
            self.logger.error(f"Error retrieving embedding: {str(e)}")
            return None

        finally:
            # Return the API key back to the queue
            if headers.get("Authorization"):
                api_key = headers["Authorization"].split("Bearer ")[-1]
                self.headers_cycle = cycle([
                    {
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json"
                    } for key in self.openai_api_keys
                ])

    def process_chunk(self, chunk_info):
        """
        Process a specific text chunk by embedding it.

        Args:
            chunk_info (tuple): A tuple containing:
                - txt_file_path (Path): Path to the text file.
                - chunk_id (int): Identifier for the chunk.
                - chunk (str): The text content of the chunk.
                - document_name (str): Name of the document.
                - page_num (int): Page number within the document.

        Returns:
            tuple: A tuple containing the embedding and its metadata if successful, (None, None) otherwise.
        """
        try:
            txt_file_path, chunk_id, chunk, document_name, page_num = chunk_info

            with self.lock:
                headers = next(self.headers_cycle)

            embedding = self.get_embedding(chunk, headers, document_name, page_num, chunk_id)

            if embedding:
                metadata = {
                    "filename": txt_file_path.name,
                    "chunk_id": chunk_id,
                    "page_num": page_num,
                    "text": chunk
                }
                return embedding, metadata

            return None, None

        except Exception as e:
            self.logger.error(f"Error processing chunk: {str(e)}")
            return None, None

    def process_file(self, txt_file_path):
        """
        Prepare chunk information from a single text file for embedding.

        Args:
            txt_file_path (Path): Path to the text file.

        Returns:
            list: A list of tuples containing chunk information for processing.
        """
        try:
            self.logger.info(f"📂 Processing file: {txt_file_path}")

            with open(txt_file_path, 'r', encoding='utf-8') as file:
                full_text = file.read()

            chunks = self.chunk_text(full_text)

            # Extract page number from filename using regex (adjust as needed)
            # Example filename: "document_page_1.txt"
            match = re.search(r'_page_(\d+)', txt_file_path.stem)
            if match:
                page_num = int(match.group(1))
            else:
                page_num = 1  # Default if not found

            chunk_infos = [
                (txt_file_path, i, chunk, txt_file_path.stem, page_num)
                for i, chunk in enumerate(chunks, 1)
            ]

            return chunk_infos

        except Exception as e:
            self.logger.error(f"Error processing file {txt_file_path}: {str(e)}")
            return []

    def process_all_files(self):
        """
        Process all text files in the input directory to generate embeddings.

        The method iterates through each text file, prepares chunks, and processes
        them concurrently using a thread pool. It aggregates all embeddings and
        associated metadata, saving them to designated output files.
        """
        try:
            txt_files = list(self.input_dir.glob('*.txt'))
            total_files = len(txt_files)
            self.logger.info(f"📢 Starting processing of {total_files} files in '{self.input_dir}'")

            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for txt_file_path in txt_files:
                    chunk_infos = self.process_file(txt_file_path)
                    for chunk_info in chunk_infos:
                        futures.append(executor.submit(
                            self.process_chunk, chunk_info
                        ))

                for future in as_completed(futures):
                    embedding, metadata = future.result()
                    if embedding and metadata:
                        self.all_embeddings.append(embedding)
                        self.all_metadata.append(metadata)

            if self.all_embeddings:
                # Save results
                chunks_json_path = self.output_dir / "chunks.json"
                with open(chunks_json_path, 'w', encoding='utf-8') as json_file:
                    json.dump({"metadata": self.all_metadata}, json_file, ensure_ascii=False, indent=4)

                embeddings_npy_path = self.output_dir / "embeddings.npy"
                np.save(embeddings_npy_path, np.array(self.all_embeddings))

                self.logger.info(f"✅ Files created: {chunks_json_path} and {embeddings_npy_path}")

        except Exception as e:
            self.logger.error(f"Error processing files: {str(e)}")
            raise
