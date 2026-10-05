import os
import requests
import re
import math
import csv
from bs4 import BeautifulSoup
import time
from lxml import html
from typing import List, Dict, Optional


class ArxivCrawlerIntegrated:
    """Integrated Arxiv paper crawler manager."""

    def __init__(self, output_dir: str = "./paper_results"):
        """Initialize the crawler.

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        self.all_papers = []  # Store all crawled papers
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_search_query(self, user_question: str) -> str:
        """Generate a search query based on the user's question.

        Args:
            user_question: The user's question or keywords

        Returns:
            The generated search query string
        """
        # Simple keyword extraction (can be replaced with a more sophisticated NLP method)
        keywords = re.findall(r'\b[a-zA-Z]{4,}\b', user_question)
        if keywords:
            # Take the first 3 keywords
            main_keywords = keywords[:3]
            query = " OR ".join(main_keywords)
        else:
            # Default query
            query = "multimodal self-growing system"

        return query

    def build_search_url(self, query: str, start: int = 0, size: int = 50) -> str:
        """Build the search URL.

        Args:
            query: The search query
            start: Starting position
            size: Page size

        Returns:
            The complete search URL
        """
        encoded_query = requests.utils.quote(query)
        return f"https://arxiv.org/search/?query={encoded_query}&searchtype=abstract&abstracts=show&order=-announced_date_first&size={size}&start={start}"

    def get_total_results(self, url: str) -> int:
        """Get the total number of results.

        Args:
            url: The search URL

        Returns:
            The total number of results
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            tree = html.fromstring(response.content)
            result_string = ''.join(tree.xpath('//*[@id="main-container"]/div[1]/div[1]/h1/text()')).strip()
            match = re.search(r'of ([\d,]+) results', result_string)
            return int(match.group(1).replace(',', '')) if match else 0
        except Exception as e:
            print(f"Failed to get the total number of results: {e}")
            return 0

    def fetch_paper_info(self, url: str) -> List[Dict]:
        """Fetch paper information for a single page.

        Args:
            url: The page URL

        Returns:
            List of paper information
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            papers = []

            for article in soup.find_all('li', class_='arxiv-result'):
                try:
                    title = article.find('p', class_='title').text.strip()
                    authors_text = article.find('p', class_='authors').text.replace('Authors:', '').strip()
                    authors = [author.strip() for author in authors_text.split(',')]
                    abstract = article.find('span', class_='abstract-full').text.strip()
                    submitted = article.find('p', class_='is-size-7').text.strip()
                    submission_date = submitted.split(';')[0].replace('Submitted', '').strip()

                    pdf_link_element = article.find('a', string='pdf')
                    pdf_link = pdf_link_element['href'] if pdf_link_element else 'No PDF link found'

                    papers.append({
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'submission_date': submission_date,
                        'pdf_link': pdf_link
                    })
                except Exception as e:
                    print(f"Failed to parse the information of a single paper: {e}")
                    continue

            return papers
        except Exception as e:
            print(f"Failed to fetch paper information: {e}")
            return []

    def crawl_papers(self, user_question: str, max_pages: int = 5) -> List[Dict]:
        """Crawl relevant papers based on the user's question.

        Args:
            user_question: The user's question or keywords
            max_pages: Maximum number of pages to crawl

        Returns:
            List of paper information
        """
        print("Searching for relevant papers based on the question...")

        # Generate the search query
        search_query = self.generate_search_query(user_question)
        print(f"Generated search query: {search_query}")

        base_url = self.build_search_url(search_query)
        total_results = self.get_total_results(base_url)

        if total_results == 0:
            print("No relevant papers found, trying the default query...")
            search_query = "multimodal self-growing system"
            base_url = self.build_search_url(search_query)
            total_results = self.get_total_results(base_url)
            if total_results == 0:
                print("No papers were found with the default query either")
                return []

        total_pages = min(math.ceil(total_results / 50), max_pages)
        self.all_papers = []

        for page in range(total_pages):
            start = page * 50
            print(f"Crawling page {page + 1}/{total_pages}, starting position: {start}")

            page_url = self.build_search_url(search_query, start=start)
            papers = self.fetch_paper_info(page_url)
            self.all_papers.extend(papers)

            time.sleep(2)  # Polite delay

        print(f"Crawl complete! Retrieved {len(self.all_papers)} relevant papers in total")
        return self.all_papers

    def save_to_csv(self, papers: List[Dict] = None, filename: str = "paper_result.csv") -> bool:
        """Save paper information to a CSV file.

        Args:
            papers: List of papers; defaults to self.all_papers
            filename: File name

        Returns:
            Whether the save succeeded
        """
        if papers is None:
            papers = self.all_papers

        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['title', 'authors', 'abstract', 'submission_date', 'pdf_link']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for paper in papers:
                    writer.writerow(paper)
            print(f"Paper information has been saved to {filepath}")
            return True
        except Exception as e:
            print(f"Failed to save the CSV file: {e}")
            return False

    def read_csv(self, filename: str) -> List[Dict]:
        """Read paper information from a CSV file.

        Args:
            filename: File name

        Returns:
            List of paper information
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            papers = []
            with open(filepath, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    title = row['title']
                    submission = row['submission_date']
                    pdf_link = row['pdf_link']
                    papers.append({
                        'title': title,
                        'submission_date': submission,
                        'pdf_link': pdf_link
                    })
            return papers
        except Exception as e:
            print(f"Failed to read the CSV file: {e}")
            return []

    def extract_year(self, submission: str) -> str:
        """Extract the year from the submission date.

        Args:
            submission: Submission date string

        Returns:
            Year string
        """
        match = re.search(r'\d{4}', submission)
        return match.group(0) if match else 'Unknown'

    def format_paper(self, paper: Dict) -> str:
        """Format the information of a single paper.

        Args:
            paper: Paper information dictionary

        Returns:
            The formatted string
        """
        title = paper['title']
        year = self.extract_year(paper['submission_date'])
        pdf_link = paper['pdf_link']
        return f"+ {title}, arxiv {year}, [[paper]]({pdf_link})."

    def generate_paper_list(self, filename: str) -> List[str]:
        """Generate a formatted paper list from a CSV file.

        Args:
            filename: CSV file name

        Returns:
            The formatted paper list
        """
        papers = self.read_csv(filename)
        return [self.format_paper(paper) for paper in papers]

    def save_formatted_papers(self, papers: List[str] = None,
                              filename: str = "formatted_papers.txt") -> bool:
        """Save the formatted paper list to a file.

        Args:
            papers: The paper list; defaults to the formatted result of self.all_papers
            filename: Output file name

        Returns:
            Whether the save succeeded
        """
        if papers is None:
            papers = self.generate_paper_list("paper_result.csv")

        try:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                for paper in papers:
                    f.write(paper + '\n')
            print(f"The formatted paper list has been saved to {filepath}")
            return True
        except Exception as e:
            print(f"Failed to save the formatted papers: {e}")
            return False

    def extract_papers_from_file(self, filename: str) -> List[Dict]:
        """Extract paper information from a text file.

        Args:
            filename: File name

        Returns:
            List of paper information
        """
        try:
            filepath = os.path.join(self.output_dir, filename)
            paper_data = []
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
                # Match pattern: title and paper link
                pattern = r'\+\s(.+?),\s(?:arxiv|arXiv)\s\d{4},\s\[\[paper\]\]\((https://arxiv\.org/pdf/[^)]+)\)\.'
                matches = re.findall(pattern, content)
                for title, paper_link in matches:
                    paper_data.append({
                        'title': title.strip(),
                        'paper_link': paper_link
                    })
            print(f"Successfully extracted {len(paper_data)} papers from the file")
            return paper_data
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            return []

    def _clean_filename(self, filename: str) -> str:
        """Clean a file name.

        Args:
            filename: The original file name

        Returns:
            The cleaned file name
        """
        illegal_chars = r'[<>:"/\\|?*]'
        clean_name = re.sub(illegal_chars, '', filename)
        clean_name = re.sub(r'\s+', ' ', clean_name).strip()
        # Limit the file name length
        if len(clean_name) > 80:
            clean_name = clean_name[:80]
        return clean_name

    def download_paper(self, paper_link: str, filepath: str) -> bool:
        """Download a single paper file.

        Args:
            paper_link: The paper PDF link
            filepath: Save path

        Returns:
            Whether the download succeeded
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            # Set the timeout
            response = requests.get(paper_link, headers=headers, timeout=30)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                print(f"Download failed, status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Download error: {e}")
            return False

    def download_papers(self, papers: List[Dict] = None,
                        max_downloads: int = 10,
                        source: str = "memory") -> int:
        """Download paper PDFs in batch.

        Args:
            papers: List of papers; defaults to self.all_papers
            max_downloads: Maximum number of downloads
            source: Paper source, either "memory" or "file" (read from the formatted file)

        Returns:
            Number of papers successfully downloaded
        """
        if papers is None:
            if source == "file":
                papers = self.extract_papers_from_file("formatted_papers.txt")
            else:
                papers = self.all_papers[:max_downloads]
        else:
            papers = papers[:max_downloads]

        if not papers:
            print("No papers to download")
            return 0

        print(f"Preparing to download {len(papers)} papers")
        success_count = 0

        for i, paper in enumerate(papers, 1):
            # Determine the paper link field name
            paper_link = paper.get('pdf_link') or paper.get('paper_link')
            if not paper_link or paper_link == 'No PDF link found':
                print(f"Skipping paper without a PDF link: {paper['title']}")
                continue

            print(f"\n[{i}/{len(papers)}] Downloading: {paper['title']}")

            # Clean the file name
            clean_title = self._clean_filename(paper['title'])
            filename = f"{clean_title}.pdf"
            filepath = os.path.join(self.output_dir, filename)

            # Check whether the file already exists
            if os.path.exists(filepath):
                print(f"File already exists, skipping: {filename}")
                success_count += 1
                continue

            # Download the paper
            start_time = time.time()
            if self.download_paper(paper_link, filepath):
                end_time = time.time()
                download_time = end_time - start_time
                print(f"✓ Download succeeded: {filename} (elapsed: {download_time:.2f}s)")
                success_count += 1
            else:
                print(f"✗ Download failed: {filename}")

            time.sleep(1)  # Download interval

        print(f"\nDownload complete! Successfully downloaded {success_count}/{len(papers)} papers")
        return success_count

    def run_full_workflow(self, user_question: str,
                          max_pages: int = 5,
                          max_downloads: int = 10) -> Dict[str, any]:
        """Run the complete workflow: crawl -> format -> download.

        Args:
            user_question: The user's question or keywords
            max_pages: Maximum number of pages to crawl
            max_downloads: Maximum number of downloads

        Returns:
            A dictionary containing the results of each step
        """
        result = {}

        # Step 1: crawl papers
        papers = self.crawl_papers(user_question, max_pages)
        result['crawled_papers'] = len(papers)

        # Save to CSV
        if papers:
            self.save_to_csv(papers)

            # Step 2: format papers
            formatted_papers = self.generate_paper_list("paper_result.csv")
            self.save_formatted_papers(formatted_papers)
            result['formatted_papers'] = len(formatted_papers)

            # Step 3: download papers
            success_downloads = self.download_papers(max_downloads=max_downloads)
            result['downloaded_papers'] = success_downloads

        return result


def create_arxiv_crawler_integrated(output_dir: str = "./paper_results") -> ArxivCrawlerIntegrated:
    """Create an integrated Arxiv crawler instance.

    Args:
        output_dir: Output directory path

    Returns:
        ArxivCrawlerIntegrated instance
    """
    return ArxivCrawlerIntegrated(output_dir)


# Usage example
if __name__ == "__main__":
    # Create a crawler instance
    crawler = create_arxiv_crawler_integrated("./my_papers")

    # Method 1: run the full workflow
    result = crawler.run_full_workflow(
        user_question="multimodal self-growing system",
        max_pages=2,
        max_downloads=5
    )
    print(f"Full workflow result: {result}")

    # Method 2: execute step by step
    # 1. Crawl papers
    # papers = crawler.crawl_papers("machine learning", max_pages=3)
    #
    # 2. Save to CSV
    # crawler.save_to_csv(papers, "ml_papers.csv")
    #
    # 3. Format papers
    # formatted = crawler.generate_paper_list("ml_papers.csv")
    # crawler.save_formatted_papers(formatted, "formatted_ml_papers.txt")
    #
    # 4. Download papers
    # success = crawler.download_papers(max_downloads=3)
    # print(f"Successfully downloaded {success} papers")
