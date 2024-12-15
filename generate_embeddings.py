import os
import logging
from pathlib import Path
from embedding_processor import EmbeddingProcessor  # Assurez-vous que la classe est dans 'embedding_processor.py'

def setup_logging():
    """
    Configure le système de logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("embedding_processor.log"),
            logging.StreamHandler()
        ]
    )

def main():
    # Configuration des chemins
    input_dir = "chemin/vers/dossier_txt"      # Remplacez par le chemin de votre dossier de fichiers .txt
    output_dir = "chemin/vers/dossier_embeddings"  # Remplacez par le chemin où vous souhaitez enregistrer les embeddings
    api_keys_file = "api_keys.txt"             # Chemin vers le fichier contenant vos clés API OpenAI

    # Vérifiez que le fichier des clés API existe
    if not Path(api_keys_file).is_file():
        print(f"🚫 Le fichier des clés API '{api_keys_file}' n'existe pas.")
        return

    # Lire les clés API depuis le fichier
    with open(api_keys_file, 'r') as f:
        openai_api_keys = [line.strip() for line in f if line.strip()]

    if not openai_api_keys:
        print("🚫 Aucune clé API OpenAI trouvée dans le fichier.")
        return

    # Configurer le logging
    setup_logging()
    logger = logging.getLogger('EmbeddingProcessor')

    # Initialiser le processeur d'embeddings
    processor = EmbeddingProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        openai_api_keys=openai_api_keys,
        verbose=True,
        logger=logger
    )

    # Exécuter le traitement des fichiers
    processor.process_all_files()
    print("🎉 Génération des embeddings terminée.")

if __name__ == "__main__":
    main()
