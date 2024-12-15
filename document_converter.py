from docx2pdf import convert
from pathlib import Path
import logging

class DocumentConverter:
    """
    Classe pour convertir des fichiers .doc et .docx en PDF.
    
    Attributes:
        input_dir (Path): Répertoire contenant les fichiers Word à convertir.
        output_dir (Path): Répertoire où les fichiers PDF convertis seront enregistrés.
        logger (logging.Logger): Logger pour enregistrer les informations et les erreurs.
    """

    def __init__(self, input_dir, output_dir, logger=None):
        """
        Initialise le DocumentConverter avec les répertoires d'entrée et de sortie.
        
        Args:
            input_dir (str ou Path): Répertoire contenant les fichiers Word.
            output_dir (str ou Path): Répertoire pour enregistrer les fichiers PDF.
            logger (logging.Logger, optional): Logger pour enregistrer les logs. Defaults à None.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

    def convert_documents(self):
        """
        Convertit tous les fichiers .doc et .docx dans le répertoire d'entrée en PDF et les enregistre dans le répertoire de sortie.
        """
        try:
            # Recherche des fichiers .doc et .docx
            doc_files = list(self.input_dir.glob('*.doc')) + list(self.input_dir.glob('*.docx'))
            total_files = len(doc_files)
            self.logger.info(f"📢 Début de la conversion de {total_files} fichiers.")

            for doc_file in doc_files:
                try:
                    self.logger.info(f"🔄 Conversion de {doc_file.name} en PDF.")
                    # Définir le chemin complet pour le fichier PDF de sortie
                    output_pdf = self.output_dir / (doc_file.stem + '.pdf')
                    convert(doc_file, output_pdf)
                    self.logger.info(f"✅ Conversion réussie: {output_pdf.name}")
                except Exception as e:
                    self.logger.error(f"❌ Erreur lors de la conversion de {doc_file.name}: {str(e)}")

            self.logger.info("🎉 Toutes les conversions sont terminées.")

        except Exception as e:
            self.logger.error(f"❌ Erreur lors de la recherche des fichiers: {str(e)}")
