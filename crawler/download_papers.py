import os
import requests
import re
import time

# Target folder
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")


# Extract paper information from a text file
def extract_papers_from_file(file_path):
    """Extract paper information from a text file."""
    paper_data = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Match pattern: title and paper link
        pattern = r'\+\s(.+?),\s(?:arxiv|arXiv)\s\d{4},\s\[\[paper\]\]\((https://arxiv\.org/pdf/[^)]+)\)'

        matches = re.findall(pattern, content)

        for title, paper_link in matches:
            paper_data.append({
                'title': title.strip(),
                'paper_link': paper_link
            })

        print(f"Successfully extracted {len(paper_data)} papers from the file")

    except FileNotFoundError:
        print(f"Error: file not found {file_path}")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    return paper_data


# Clean the file name
def clean_filename(filename):
    """Clean the file name by removing characters that are not allowed on Windows."""
    illegal_chars = r'[<>:"/\\|?*]'
    clean_name = re.sub(illegal_chars, '', filename)
    clean_name = re.sub(r'\s+', ' ', clean_name).strip()
    # Limit the file name length
    if len(clean_name) > 80:
        clean_name = clean_name[:80]
    return clean_name


# Download file function
def download_paper(paper_link, filepath):
    """Download a single paper file."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
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


# Main program
def main():
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created: {output_dir}")

    # Read the paper text file
    file_path = "formatted_papers.txt"

    # Extract paper information
    print("Extracting paper information from the file...")
    all_papers = extract_papers_from_file(file_path)

    if not all_papers:
        print("No paper information found. Please check the file path and content format")
        return

    # Only take the first 10 papers
    papers_to_download = all_papers[:10]

    print(f"Found {len(all_papers)} papers; the first {len(papers_to_download)} will be downloaded")
    print("=" * 80)

    # Show the information of the first 10 papers
    print("List of the first 10 papers:")
    for i, paper in enumerate(papers_to_download, 1):
        print(f"{i:2d}. {paper['title']}")
    print("=" * 80)

    # Download papers sequentially
    success_count = 0

    for i, paper in enumerate(papers_to_download, 1):
        print(f"\n[{i}/{len(papers_to_download)}] Downloading: {paper['title']}")

        # Clean the file name (use only the paper title, no prefix)
        clean_title = clean_filename(paper['title'])
        filename = f"{clean_title}.pdf"
        filepath = os.path.join(output_dir, filename)

        # Check whether the file already exists
        if os.path.exists(filepath):
            print(f"File already exists, skipping: {filename}")
            success_count += 1
            continue

        # Download the paper
        start_time = time.time()
        if download_paper(paper['paper_link'], filepath):
            end_time = time.time()
            download_time = end_time - start_time
            print(f"✓ Download succeeded: {filename} (elapsed: {download_time:.2f}s)")
            success_count += 1
        else:
            print(f"✗ Download failed: {filename}")

        print("-" * 60)

    # Download result summary
    print(f"\nDownload complete!")
    print(f"Successfully downloaded: {success_count}/{len(papers_to_download)} papers")
    print(f"File save location: {output_dir}")


# Run the main program
if __name__ == "__main__":
    main()
