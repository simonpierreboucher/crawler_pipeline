import logging
from pathlib import Path
from document_converter import DocumentConverter

def setup_logging(log_file='document_converter.log'):
    """
    Configure le système de logging.
    
    Args:
        log_file (str, optional): Nom du fichier de log. Defaults à 'document_converter.log'.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('DocumentConverter')
    return logger

def main():
    # Configurer le logger
    logger = setup_logging()

    # Définir les répertoires d'entrée et de sortie
    input_directory = Path('./input_documents')  # Remplacez par votre répertoire d'entrée
    output_directory = Path('./output_pdfs')     # Remplacez par votre répertoire de sortie

    # Vérifier que le répertoire d'entrée existe
    if not input_directory.exists():
        logger.error(f"📁 Le répertoire d'entrée {input_directory} n'existe pas.")
        return

    # Initialiser le convertisseur
    converter = DocumentConverter(
        input_dir=input_directory,
        output_dir=output_directory,
        logger=logger
    )

    # Exécuter la conversion
    converter.convert_documents()

if __name__ == "__main__":
    main()
