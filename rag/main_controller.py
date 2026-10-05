# main_controller.py (modified version)
import os
import time
from pdf_processor import PDFProcessor
from rag_system import setup_rag_system
# from arxiv_crawler import create_arxiv_crawler
from arxiv_crawler_integrated import create_arxiv_crawler_integrated

class OCRRAGController:
    def __init__(self, pdf_folder_path):
        self.pdf_folder_path = pdf_folder_path
        self.md_folder_path = "./md"
        self.rag_system = None
        self.last_processed_time = 0
        self.arxiv_crawler = None
        self.last_crawl_time = 0
        self.crawl_cooldown = 3600  # 1-hour cooldown to avoid frequent crawling
        self.use_enhanced = False
        self.process_pdf_folder = None

    def setup_pdf_processor(self):
        """Set up the OCR model."""
        self.process_pdf_folder = PDFProcessor(
            output_dir=self.md_folder_path,  # Output directory
            lang='en',  # Language setting ('ch' or 'en')
            dpi=220  # OCR resolution
        )

    def setup_arxiv_crawler(self):
        """Set up the Arxiv crawler."""
        # self.arxiv_crawler = create_arxiv_crawler(self.pdf_folder_path)
        self.arxiv_crawler = create_arxiv_crawler_integrated(self.pdf_folder_path)

    def fetch_latest_papers(self,question, max_papers=5):
        """Fetch and download the latest papers."""
        if self.arxiv_crawler is None:
            self.setup_arxiv_crawler()

        # Check the cooldown
        current_time = time.time()
        if current_time - self.last_crawl_time < self.crawl_cooldown:
            print("The crawler is cooling down, please try again later")
            return False

        try:
            print("=" * 60)
            print("Starting to fetch the latest papers from Arxiv...")
            print("=" * 60)

            # Crawl paper information
            papers = self.arxiv_crawler.crawl_papers(question,max_pages=2)
            if not papers:
                print("No paper information was retrieved")
                return False
            # self.arxiv_crawler.save_to_csv(papers)  # Save to CSV
            # self.arxiv_crawler.save_formatted_papers()  # Save to TXT
            # Download the paper PDFs
            success_count = self.arxiv_crawler.download_papers(
                papers=papers,
                max_downloads=max_papers
            )

            if success_count > 0:
                print(f"Successfully downloaded {success_count} new papers")
                self.last_crawl_time = current_time
                return True
            else:
                print("No papers were successfully downloaded")
                return False

        except Exception as e:
            print(f"Failed to fetch papers: {e}")
            return False

    def check_for_pdfs_and_mds(self):
        """Check whether there are new PDF files that need processing."""
        pdf_folder_exit = True
        md_folder_exit = True
        if not os.path.exists(self.pdf_folder_path):
            pdf_folder_exit = False
            print(f"PDF folder does not exist: {self.pdf_folder_path}")
        if not os.path.exists(self.md_folder_path):
            md_folder_exit = False
            print(f"MD folder does not exist: {self.md_folder_path}")
        if not pdf_folder_exit or not md_folder_exit:
            return [],[]

        pdf_files = [f for f in os.listdir(self.pdf_folder_path)
                     if f.lower().endswith('.pdf')]
        md_files = [f for f in os.listdir(self.md_folder_path)
                     if f.lower().endswith('.md')]
        return pdf_files,md_files

    def process_all_pdfs(self):
        """Process all PDF files and update the RAG system."""
        print("Starting to process the PDF folder...")

        # Process the PDF folder
        processed_files = self.process_pdf_folder.process_pdf_folder(self.pdf_folder_path)

        if processed_files:
            print("PDF processing complete, updating the RAG system...")
            # Re-initialize the RAG system
            self.rag_system.update_rag_system()
            self.last_processed_time = time.time()
            print("RAG system update complete!")
        else:
            print("No processable PDF files were found")

        return processed_files


    ## Needs improvement: relevance improvement
    def evaluate_relevance(self, retrieved_docs):
        """Evaluate the relevance of the retrieved documents to the question."""
        if not retrieved_docs:
            return False, "No relevant documents were retrieved"

        # Simple evaluation: check document count and quality
        relevant_count = 0
        total_content_length = 0

        for doc in retrieved_docs:
            content = doc.page_content
            # Simple keyword matching (can be replaced with more complex semantic matching)
            if len(content) > 50:  # Ensure the document has enough content
                total_content_length += len(content)
                relevant_count += 1

        # Judgment criteria: at least 1 relevant document and sufficient total content length
        is_relevant = relevant_count >= 1 and total_content_length > 200

        reason = f"Retrieved {relevant_count} relevant documents with a total content length of {total_content_length} characters"

        return is_relevant, reason

    def ask_question_with_fallback(self, question, show_docs=True, progress_callback=None):
        """Question answering with a fallback mechanism: if relevance is insufficient, automatically trigger the crawler.

        Args:
            question: The user's question
            show_docs: Whether to print the retrieved documents
            progress_callback: Optional callback for reporting workflow progress (used by the API layer to write to the timeline)
        """
        def report(message: str):
            """Report progress uniformly: print to the console and also call back to the API layer."""
            print(message)
            if progress_callback:
                progress_callback(message)

        if self.rag_system is None:
            report("RAG system is not initialized")
            return "The system is not ready. Please initialize the RAG system first.", []
        try:
            # First attempt: answer using the existing knowledge base
            report("Searching the existing knowledge base...")
            answer, relevant_docs = self.rag_system.ask_question(
                question,
                show_docs=show_docs,
                use_enhanced=self.use_enhanced
            )

            # Evaluate relevance
            is_relevant, reason = self.evaluate_relevance(relevant_docs)
            if is_relevant:
                report(f"Document relevance evaluation: passed ({reason})")
                return answer, relevant_docs
            else:
                report(f"Document relevance evaluation: insufficient ({reason})")
                report("Searching online and downloading the latest papers...")

                # Trigger the crawler to fetch new papers
                if self.fetch_latest_papers(question, max_papers=3):
                    # Process the new papers and update the RAG system
                    report("Extracting text via OCR and updating the knowledge base...")
                    self.process_all_pdfs()

                    # Second attempt: answer using the updated knowledge base
                    report("Re-answering the question using the updated knowledge base...")
                    answer, docs = self.rag_system.ask_question(
                        question,
                        show_docs=show_docs,
                        use_enhanced=self.use_enhanced
                    )

                    # Evaluate again
                    is_relevant, reason = self.evaluate_relevance(docs)
                    if is_relevant:
                        report(f"Updated document relevance evaluation: passed ({reason})")
                    else:
                        report(f"Updated document relevance evaluation: still insufficient ({reason})")
                        answer = f"{answer}\n\nNote: even after fetching the latest papers, the relevant information is still limited."

                    return answer, docs
                else:
                    return "Could not fetch the latest papers. Please check your network connection or try again later.", []

        except Exception as e:
            report(f"An error occurred while processing the question: {e}")
            return f"An error occurred while processing the question: {e}", []

    def run_interactive(self):
        """Run the interactive question-answering system."""
        # Initialize the crawler
        self.setup_arxiv_crawler()
        # Initialize OCR
        self.setup_pdf_processor()
        # Check the PDF folder status
        pdf_files,md_files = self.check_for_pdfs_and_mds()
        if not pdf_files and not md_files:
            print("No PDF or MD files found. Need to fetch an initial paper library from Arxiv")
            self.fetch_latest_papers("multimodal self-growing system", max_papers=5)
        elif not pdf_files:
            print(f"The PDF folder is empty, but {len(md_files)} MD files were found. Existing MD files will be used")
        elif not md_files:
            print(f"The MD folder is empty, but {len(pdf_files)} PDF files were found. These PDFs will be processed")


        # Initialize the RAG system
        if self.rag_system is None:
            print("Initializing the RAG system...")
            self.rag_system = setup_rag_system()
            self.last_processed_time = time.time()

        if self.rag_system is None:
            print("RAG system initialization failed. Unable to start question answering")
            return

        print("\n" + "=" * 60)
        print("Type 'exit' to end the program")
        # print("Type 'manual update' to force fetching the latest papers")
        # print("Type 'refresh' to re-process the existing PDF files")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nPlease enter a question: ").strip()

                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("Thanks for using!")
                    break
                # elif user_input.lower() in ['manual update', 'update']:
                #     print("Starting manual paper library update...")
                #     if self.fetch_latest_papers(max_papers=5):
                #         self.process_all_pdfs()
                #     continue
                # elif user_input.lower() in ['refresh', 'reload']:
                #     print("Re-processing PDF files...")
                #     self.process_all_pdfs()
                #     continue
                # elif user_input.lower() in ['mode']:
                #     enhanced_mode = not enhanced_mode
                #     mode_name = "Enhanced mode" if enhanced_mode else "Simple mode"
                #     print(f"Switched to {mode_name}")
                #     continue

                if not user_input:
                    continue

                # Use the intelligent question-answering system (auto-triggers the crawler fallback mechanism)
                print("Analyzing the question and generating an answer...")
                answer, docs = self.ask_question_with_fallback(user_input, show_docs=True)
                print(f"Answer:\n{answer}")

            except KeyboardInterrupt:
                print("\nProgram interrupted by the user")
                break
            except Exception as e:
                print(f"An error occurred while processing the question: {e}")


def main():
    # Configure the PDF folder path
    PDF_FOLDER = "./pdf"

    # Ensure the PDF folder exists
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs("./documents", exist_ok=True)

    # Create the controller
    controller = OCRRAGController(PDF_FOLDER)

    # Start the system
    controller.run_interactive()


if __name__ == "__main__":
    main()
