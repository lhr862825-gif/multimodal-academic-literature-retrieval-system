import csv
import re

def read_csv(file_name):
    papers = []
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            title = row['title']
            submission = row['submission_date']
            pdf_link = row['pdf_link']
            papers.append({
            'title': title, 'submission_date': submission, 'pdf_link': pdf_link})
    return papers

def extract_year(submission):

    match = re.search(r'\d{4}', submission)
    if match:
        return match.group(0)
    else:
        return 'Unknown'

def format_paper(paper):
    title = paper['title']

    year = extract_year(paper['submission_date'])
    pdf_link = paper['pdf_link']
    return f"+ {
              title}, arxiv {
              year}, [[paper]]({
              pdf_link})."

def generate_paper_list(file_name):
    papers = read_csv(file_name)
    formatted_papers = [format_paper(paper) for paper in papers]
    return formatted_papers

def save_to_file(papers, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for paper in papers:
            f.write(paper + '\n')

# Main program
input_file = 'paper_result.csv'  # Input CSV file
output_file = 'formatted_papers.txt'  # Output file for the formatted text
papers = generate_paper_list(input_file)
save_to_file(papers, output_file)

print(f"Saved {
              len(papers)} formatted paper records to {
              output_file}.")
