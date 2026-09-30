from pypdf import PdfReader
import os
import glob

pdf_folder = "data/raw"  # folder containing all your PDFs
pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))

for pdf_path in pdf_paths:
    print(f"Processing {pdf_path}...")
    # your existing chunking logic here, using pdf_path
    reader = PdfReader(pdf_path)

    for i, page in enumerate(reader.pages):
       text = page.extract_text()
       if text:
         print(f"page{i+1}:{len(text)} characters")
       else:
         print(f"page {i+1}: NO TEXT")

