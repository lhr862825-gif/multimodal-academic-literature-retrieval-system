import argparse
import time
import os
import re
import requests
from langchain_community.retrievers import ArxivRetriever

# ================= Configuration =================
SUMMARY_DIR = "arxiv_summary_notes"
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "papers")


def setup_directories():
    """Create the required folders."""
    for d in [SUMMARY_DIR, PDF_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"📁 Folder created: {d}")


def sanitize_filename(name):
    """Clean the file name."""
    name = re.sub(r'\s+', '_', name)
    return re.sub(r'[\\/*?:"<>|]', "", name)


def extract_arxiv_id(entry_id):
    """Extract the arXiv ID from the Entry ID."""
    if not entry_id:
        return None

    entry_str = str(entry_id)

    patterns = [
        r'arxiv\.org/abs/(\d+\.\d+)',
        r'arxiv\.org/pdf/(\d+\.\d+)',
        r'arXiv:(\d+\.\d+)',
        r'arxiv\.org/abs/([a-z\-]+/\d+\.\d+)',
        r'arxiv\.org/pdf/([a-z\-]+/\d+\.\d+)',
        r'arXiv:([a-z\-]+/\d+\.\d+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, entry_str)
        if match:
            arxiv_id = match.group(1)
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]
            return arxiv_id

    return None


def download_pdf(arxiv_id, title):
    """Download the PDF file."""
    if not arxiv_id:
        return False, None

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    clean_arxiv_id = arxiv_id.replace('/', '_')
    safe_title = sanitize_filename(title)[:80]
    filename = f"{clean_arxiv_id}_{safe_title}.pdf"
    filepath = os.path.join(PDF_DIR, filename)

    print(f"   📥 Downloading PDF: {arxiv_id}")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf"
        }

        response = requests.get(pdf_url, headers=headers, timeout=60)

        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)

            file_size = os.path.getsize(filepath) / 1024
            print(f"   ✅ PDF download complete ({file_size:.1f}KB)")
            return True, filepath
        else:
            print(f"   ❌ HTTP error {response.status_code}")
            return False, None

    except Exception as e:
        print(f"   ❌ Download failed: {str(e)}")
        return False, None


def save_summary(doc, query, download_pdf_option=False):
    """Save the summary note."""
    safe_query = sanitize_filename(query)

    # Get the metadata
    title = doc.metadata.get('Title', 'No Title').replace('\n', ' ')
    authors = doc.metadata.get('Authors', 'Unknown')
    published = doc.metadata.get('Published', 'Unknown')
    entry_id = doc.metadata.get('Entry ID', '')

    # Extract the arXiv ID
    arxiv_id = extract_arxiv_id(entry_id)

    safe_title = sanitize_filename(title)[:50]
    filename = f"{safe_query}_{safe_title}_summary.md"
    filepath = os.path.join(SUMMARY_DIR, filename)

    # PDF download
    pdf_saved = False
    pdf_path = ""
    if download_pdf_option and arxiv_id:
        pdf_success, pdf_path = download_pdf(arxiv_id, title)
        pdf_saved = pdf_success

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# ArXiv Summary Note: {title}\n\n")
        f.write(f"- **Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Authors:** {authors}\n")
        f.write(f"- **Published:** {published}\n")

        # ========== Display of the original Entry ID ==========
        f.write(f"\n**📌 Original Entry ID (retrieved directly from ArxivRetriever):**\n")
        if entry_id:
            f.write(f"> `{entry_id}`\n\n")

            # Show basic information about the Entry ID
            f.write(f"**Entry ID analysis:**\n")
            f.write(f"- **Length:** {len(entry_id)} characters\n")
            f.write(f"- **Type:** `{type(entry_id).__name__}`\n")
        else:
            f.write(f"> `(empty string or no Entry ID)` ⚠️\n")

        # arXiv identifier
        f.write(f"\n**📌 Paper identifier:**\n")
        if arxiv_id:
            f.write(f"- **arXiv ID:** `{arxiv_id}`\n")
            f.write(f"- **Abstract link:** [arxiv.org/abs/{arxiv_id}](https://arxiv.org/abs/{arxiv_id})\n")
            f.write(f"- **PDF link:** [arxiv.org/pdf/{arxiv_id}.pdf](https://arxiv.org/pdf/{arxiv_id}.pdf)\n")
        else:
            f.write(f"- **arXiv ID:** not found\n")
            f.write(f"- **Original Entry ID:** {entry_id[:100] if entry_id else 'none'}\n")

        # PDF status
        if download_pdf_option:
            f.write(f"\n**📂 PDF download status:**\n")
            if pdf_saved and pdf_path:
                abs_path = os.path.abspath(pdf_path)
                file_size = os.path.getsize(pdf_path) / 1024
                f.write(f"- ✅ Downloaded: `{os.path.basename(pdf_path)}` ({file_size:.1f}KB)\n")
                f.write(f"- 📍 Path: `{abs_path}`\n")
            else:
                f.write(f"- ❌ Download failed\n")

        f.write("\n---\n\n")
        f.write("### 📝 Abstract\n\n")
        f.write(f"> {doc.page_content}\n")

        # Statistics
        char_count = len(doc.page_content)
        word_count = len(doc.page_content.split())
        f.write(f"\n*Statistics: {word_count} words, {char_count} characters*\n")

    return filepath, pdf_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--count', type=int, default=3)
    parser.add_argument('--download-pdf', action='store_true', help="Whether to download PDFs")
    args = parser.parse_args()

    setup_directories()

    print("\n" + "=" * 50)
    print("🦜 ArXiv paper abstract downloader")
    print("=" * 50)
    print(f"📁 Abstract notes: {SUMMARY_DIR}")
    print(f"📁 PDF save directory: {PDF_DIR}")
    print("=" * 50)

    # Interactive input
    if not args.query:
        args.query = input("\n👉 Search keywords: ").strip()
        if not args.query:
            print("❌ Keywords cannot be empty")
            return

    if not args.download_pdf:
        pdf_input = input("👉 Download PDFs? (y/n, default n): ").strip().lower()
        args.download_pdf = (pdf_input == 'y')

    print(f"\n🚀 Searching: '{args.query}'")
    if args.download_pdf:
        print(f"📥 PDF download: ✅ enabled")

    try:
        # Get abstracts
        retriever = ArxivRetriever(
            load_max_docs=args.count,
            get_full_documents=False
        )
        summary_docs = retriever.invoke(args.query)

        if not summary_docs:
            print("❌ No relevant papers found.")
            return

        print(f"✅ Found {len(summary_docs)} papers")

        # Save abstracts and PDFs
        pdf_count = 0
        for i, doc in enumerate(summary_docs, 1):
            title = doc.metadata.get('Title', 'No Title')[:70]
            print(f"  [{i}] Processing: {title}")

            path, pdf_success = save_summary(doc, args.query, args.download_pdf)
            if pdf_success:
                pdf_count += 1
            print(f"    Abstract saved: {os.path.basename(path)}")

        # Completion info
        print(f"\n" + "=" * 50)
        print("🎉 Done!")
        print(f"📄 Abstract notes directory: {os.path.abspath(SUMMARY_DIR)}")

        if args.download_pdf:
            print(f"📂 PDF save directory: {os.path.abspath(PDF_DIR)}")
            print(f"📥 Successfully downloaded: {pdf_count}/{len(summary_docs)} PDFs")

        print("=" * 50)

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()
