import sys
import os
import re
import shutil
import subprocess
import tempfile
import json
import fitz
from collections import Counter
import glob


class PDFProcessor:
    """PDF text extraction processor."""

    def __init__(self, output_dir="./md", lang='en', dpi=220):
        """
        Initialize the PDF processor.

        Args:
            output_dir: Output directory path
            lang: Language setting ('ch' or 'en')
            dpi: OCR recognition resolution
        """
        self.output_dir = output_dir
        self.lang = lang
        self.dpi = dpi
        self.ocr_engine = None

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _enough_text(self, s, min_len=5, alpha_ratio=0.2):
        """Check whether the text is long enough and contains enough letters/Chinese characters."""
        t = ''.join(ch for ch in s if not ch.isspace())
        if len(t) < min_len:
            return False
        a = sum(c.isalpha() or '\u4e00' <= c <= '\u9fff' for c in t) / max(1, len(t))
        return a >= alpha_ratio

    def _normalize(self, s):
        """Text normalization."""
        s = s.replace('\r', '')
        s = re.sub(r'(?<=[\u4e00-\u9fff])\s*\n(?!\n)\s*(?=[\u4e00-\u9fff])', '', s)
        s = re.sub(r'-\s*\n(?=[A-Za-z])', '', s)
        s = re.sub(r'[ \t]+', ' ', s)
        s = re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    def _pdftotext_page(self, pdf, p):
        """Extract a single page of text using the external pdftotext tool."""
        exe = shutil.which('pdftotext')
        if not exe:
            return ''
        r = subprocess.run([exe, '-enc', 'UTF-8', '-raw', '-f', str(p + 1), '-l', str(p + 1), pdf, '-'],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8', errors='ignore')
        return r.stdout or ''

    def _extract_by_layout(self, page):
        """Extract text based on layout analysis."""
        try:
            d = page.get_text('dict')
        except Exception:
            return page.get_text('text')
        h = float(page.rect.height)
        items = []
        for b in d.get('blocks', []):
            if b.get('type') != 0:
                continue
            for ln in b.get('lines', []):
                lb = ln.get('bbox') or [0, 0, 0, 0]
                x0 = float(lb[0])
                y0 = float(lb[1])
                s = ''.join(sp.get('text', '') for sp in ln.get('spans', []))
                if s.strip():
                    items.append((x0, y0, s))
        if not items:
            return ''
        items.sort(key=lambda t: (t[1], t[0]))
        thr_y = max(8.0, 0.02 * h)
        rows = []
        for x0, y0, s in items:
            placed = False
            for r in rows:
                if abs(y0 - r['y']) <= thr_y:
                    r['cells'].append((x0, s))
                    r['y'] = (r['y'] + y0) / 2.0
                    placed = True
                    break
            if not placed:
                rows.append({'y': y0, 'cells': [(x0, s)]})
        rows.sort(key=lambda r: r['y'])
        parts = []
        prev_y = None
        for r in rows:
            r['cells'].sort(key=lambda c: c[0])
            row_text = ' '.join([c[1] for c in r['cells']])
            if prev_y is not None and (r['y'] - prev_y) > (thr_y * 2.0):
                parts.append('')
            parts.append(row_text)
            prev_y = r['y']
        return '\n'.join(parts)

    def _split_sentences(self, p):
        """Split sentences."""
        return [s for s in re.split(r'(?<=[。！？；.!?;:])', p) if s.strip()]

    def _make_segments(self, pages, max_len=800):
        """Split text into paragraphs."""
        segs = []
        offset = 0
        for idx, p in enumerate(pages):
            paras = [x.strip() for x in re.split(r'\n{2,}', p) if x.strip()]
            buf = []
            for para in paras:
                for s in self._split_sentences(para):
                    if len(''.join(buf)) + len(s) > max_len and buf:
                        text = ''.join(buf)
                        segs.append((idx, offset, text))
                        offset += len(text)
                        buf = [s]
                    else:
                        buf.append(s)
            if buf:
                text = ''.join(buf)
                segs.append((idx, offset, text))
                offset += len(text)
        return segs

    def _build_ocr(self):
        """Build the OCR engine."""
        try:
            from paddleocr import PaddleOCR
        except Exception:
            return None
        base = os.environ.get('OCR_MODELS_DIR')
        det_dir = os.environ.get('OCR_DET_DIR')
        rec_dir = os.environ.get('OCR_REC_DIR')
        if base and not det_dir:
            p = os.path.join(base, 'PP-OCRv5_server_det')
            if os.path.isdir(p):
                det_dir = p
        if base and not rec_dir:
            rn = 'ch_PP-OCRv5_mobile_rec' if str(self.lang).startswith('ch') else 'en_PP-OCRv5_mobile_rec'
            p = os.path.join(base, rn)
            if os.path.isdir(p):
                rec_dir = p
            else:
                p2 = os.path.join(base, 'PP-OCRv5_server_rec')
                if os.path.isdir(p2):
                    rec_dir = p2
        kwargs = {
            'use_gpu': True,
            'gpu_id': 1,  # Specify GPU index 1
            'use_angle_cls': False,
            'lang': self.lang
        }
        if det_dir:
            kwargs['det_model_dir'] = det_dir
        if rec_dir:
            kwargs['rec_model_dir'] = rec_dir
        return PaddleOCR(**kwargs)

    def _pdf_to_text(self, pdf_path, out_md):
        """Core method for extracting text from a PDF."""
        ocr = None
        doc = fitz.open(pdf_path)
        lines_per_page = []
        texts = []
        with tempfile.TemporaryDirectory() as td:
            for i in range(doc.page_count):
                page = doc[i]
                t = self._extract_by_layout(page)
                if not self._enough_text(t):
                    t2 = self._pdftotext_page(pdf_path, i)
                    t = t2 if self._enough_text(t2) else t
                if not self._enough_text(t):
                    if ocr is None:
                        ocr = self._build_ocr()
                    pix = page.get_pixmap(dpi=self.dpi)
                    img = os.path.join(td, f'p_{i + 1}.png')
                    pix.save(img)
                    ot = []
                    res = None
                    if ocr is not None:
                        try:
                            res = ocr.ocr(img)
                        except Exception:
                            try:
                                res = ocr.predict(img)
                            except Exception:
                                res = None
                    if res:
                        try:
                            seq = res if isinstance(res, list) else list(res)
                            for line in seq:
                                if isinstance(line, dict) and 'res' in line:
                                    r = line['res']
                                    if isinstance(r, dict):
                                        if 'rec_texts' in r and isinstance(r['rec_texts'], list):
                                            ot += [x for x in r['rec_texts'] if isinstance(x, str)]
                                        elif 'rec_text' in r and isinstance(r['rec_text'], str):
                                            ot.append(r['rec_text'])
                                    continue
                                if isinstance(line, list):
                                    for item in line:
                                        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], tuple):
                                            ot.append(item[1][0])
                                        elif isinstance(item, dict) and 'text' in item:
                                            ot.append(item['text'])
                                        else:
                                            s = str(item)
                                            m = re.search(r"text='([^']+)'", s) or re.search(r'"text"\s*:\s*"([^"]+)"',
                                                                                             s)
                                            if m:
                                                ot.append(m.group(1))
                                elif isinstance(line, dict) and 'text' in line:
                                    ot.append(line['text'])
                                else:
                                    s = str(line)
                                    m = re.search(r"text='([^']+)'", s) or re.search(r'"text"\s*:\s*"([^"]+)"', s)
                                    if m:
                                        ot.append(m.group(1))
                        except Exception:
                            pass
                    t = '\n'.join(ot)
                t = self._normalize(t)
                texts.append(t)
                ls = [x for x in t.split('\n') if x.strip()]
                head = ls[:3]
                tail = ls[-3:] if len(ls) >= 3 else ls[-len(ls):]
                lines_per_page.append((head, tail))
        # doc.close()
        commons = set()
        if doc.page_count >= 3:
            cnt = Counter()
            for h, t in lines_per_page:
                for x in set(h + t):
                    cnt[x] += 1
            threshold = max(2, int(0.3 * doc.page_count))
            commons = {x for x, c in cnt.items() if c >= threshold and len(x) <= 80}
        cleaned = []
        for t in texts:
            cleaned.append('\n'.join([ln for ln in t.split('\n') if ln.strip() and ln not in commons]))
        os.makedirs(os.path.dirname(out_md), exist_ok=True)
        md_parts = []
        for i, page_text in enumerate(cleaned):
            md_parts.append(f"## Page {i + 1}\n\n{page_text}")
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(md_parts))
        # stem, ext = os.path.splitext(out_md)
        # jsonl_path = stem + '.jsonl'
        # segs = self._make_segments(cleaned)
        # with open(jsonl_path, 'w', encoding='utf-8') as jf:
        #     for page_idx, char_offset, text in segs:
        #         jf.write(
        #             json.dumps(
        #                 {"page_index": page_idx + 1, "char_offset": char_offset, "length": len(text), "text": text},
        #                 ensure_ascii=False) + '\n')

    def process_pdf_folder(self, pdf_folder_path):
        """
        Process all PDF files in the PDF folder.

        Args:
            pdf_folder_path: PDF folder path

        Returns:
            list: List of successfully processed files
        """
        if not os.path.exists(pdf_folder_path):
            raise FileNotFoundError(f"PDF folder does not exist: {pdf_folder_path}")

        # Get all PDF files
        pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
        pdf_files += glob.glob(os.path.join(pdf_folder_path, "*.PDF"))

        if not pdf_files:
            print(f"No PDF files found in {pdf_folder_path}")
            return []

        processed_files = []

        # Process each PDF file
        for pdf_file in pdf_files:
            try:
                print(f"Processing: {pdf_file}")

                # Generate the output file name
                base_name = os.path.splitext(os.path.basename(pdf_file))[0]
                out_md = os.path.join(self.output_dir, base_name + '.md')

                # Process the PDF
                self._pdf_to_text(pdf_file, out_md)

                processed_files.append(pdf_file)
                print(f"Done: {pdf_file} -> {out_md}")

            except Exception as e:
                print(f"An error occurred while processing file {pdf_file}: {str(e)}")
                continue

        return processed_files


# Usage example
if __name__ == '__main__':
    # Create a processor instance
    processor = PDFProcessor(output_dir="./md", lang='en', dpi=220)

    # Process the PDF folder
    pdf_folder = "./pdf"  # Replace with your PDF folder path
    result = processor.process_pdf_folder(pdf_folder)

    print(f"Successfully processed {len(result)} PDF files")
