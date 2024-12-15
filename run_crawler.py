import re
from web_crawler import WebCrawler  # Assurez-vous que le code WebCrawler est sauvegardé dans web_crawler.py

def main():
    # Définissez votre configuration
    start_url = "https://liquid-air.ca"  # Remplacez par l'URL que vous souhaitez crawler
    max_depth = 3
    use_playwright = True  # Mettez à False si vous ne souhaitez pas utiliser Playwright
    excluded_paths = ['selecteur-de-produits']  # Ajouter d'autres segments à exclure si nécessaire
    download_extensions = {
        'PDF': ['.pdf'],
        'Image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'],
        'Doc': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
    }
    language_pattern = re.compile(r'/(fr|en)-(ca|us)/')  # Ajustez le pattern selon la langue/culture

    # Définissez le répertoire de base pour les sorties (optionnel)
    base_dir = "crawler_output"

    # Instanciez le WebCrawler
    crawler = WebCrawler(
        start_url=start_url,
        max_depth=max_depth,
        use_playwright=use_playwright,
        excluded_paths=excluded_paths,
        download_extensions=download_extensions,
        language_pattern=language_pattern,
        base_dir=base_dir
    )

    # Lancez le crawling
    crawler.crawl()

if __name__ == "__main__":
    main()
