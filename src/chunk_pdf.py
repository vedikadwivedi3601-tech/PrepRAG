
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import glob
import re


pdf_folder = "data/raw"

# Find all PDF files
pdf_paths = glob.glob(os.path.join(pdf_folder, "*.pdf"))


# Function to clean text
def clean_text(text):

    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'Page \d+', '', text)

    return text.strip()


# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=80
)


# Store chunks from ALL PDFs
all_chunks = []


# Process every PDF
for pdf_path in pdf_paths:

    print(f"\nProcessing: {pdf_path}")

    # Load PDF
    reader = PdfReader(pdf_path)

    # Extract and clean text
    documents = []

    for page_number, page in enumerate(reader.pages):

        text = page.extract_text()

        if text:

            text = clean_text(text)

            documents.append({
                "page": page_number + 1,
                "text": text
            })

    print("Total pages with text:", len(documents))


    # Create chunks
    for document in documents:

        chunks = text_splitter.split_text(
            document["text"]
        )

        for chunk in chunks:

            all_chunks.append({
                "source": os.path.basename(pdf_path),
                "page": document["page"],
                "text": chunk
            })


print("\nTotal chunks:", len(all_chunks))


# Display first 3 chunks
for i, chunk in enumerate(all_chunks[:3]):

    print("\n" + "=" * 60)

    print("CHUNK:", i + 1)
    print("SOURCE:", chunk["source"])
    print("PAGE:", chunk["page"])

    print("=" * 60)

    print(chunk["text"])