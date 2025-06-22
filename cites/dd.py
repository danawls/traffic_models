from fpdf import FPDF
from pdf2image import convert_from_path
import pytesseract

# Helper function to filter unsupported characters


def filter_text_for_fpdf(text):
    return text.encode('latin-1', 'replace').decode('latin-1')


# Convert PDF to images
pdf_path = "newell1961.pdf"
images = convert_from_path(pdf_path)

# Create a FPDF object for saving OCR result
pdf = FPDF()

# Loop over each image and process it
for img in images:
    # Perform OCR on the image
    text = pytesseract.image_to_string(img)

    # Filter text to remove unsupported characters
    filtered_text = filter_text_for_fpdf(text)

    # Create a new page in the PDF
    pdf.add_page()

    # Add the text to the page
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, filtered_text)

# Save the resulting PDF
output_pdf_path = "converted_text_output_final.pdf"
pdf.output(output_pdf_path)
