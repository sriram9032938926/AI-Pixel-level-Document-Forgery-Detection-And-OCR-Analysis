import fitz  # PyMuPDF
import numpy as np
import cv2


def pdf_bytes_to_images(pdf_bytes, dpi=200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        images.append(img)

    doc.close()
    return images


def pdf_file_to_images(pdf_path, dpi=200):
    doc = fitz.open(pdf_path)

    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        images.append(img)

    doc.close()
    return images