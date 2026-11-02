import io
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import pypdfium2

sys.path.insert(0, os.path.abspath('backend'))
import backend
from backend import app, get_current_user, run_tesseract_ocr

class TestUploadSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_user = MagicMock()
        cls.mock_user.id = "00000000-0000-0000-0000-000000000001"
        cls.mock_user.email = "test@example.com"
        app.dependency_overrides[get_current_user] = lambda: cls.mock_user
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()

    def test_oversized_file_content_length_returns_413(self):
        """Content-Length exceeding MAX_UPLOAD_MB must immediately return 413 without loading body."""
        resp = self.client.post(
            "/upload",
            files={"file": ("large.pdf", b"small bytes", "application/pdf")},
            headers={"Content-Length": str(20 * 1024 * 1024)}  # 20 MB header
        )
        self.assertEqual(resp.status_code, 413)

    def test_fake_pdf_missing_header_returns_422(self):
        """File with .pdf extension but missing %PDF magic header returns 422."""
        fake_pdf = b"This is plain text pretending to be a PDF."
        resp = self.client.post(
            "/upload",
            files={"file": ("spoofed.pdf", fake_pdf, "application/pdf")}
        )
        self.assertEqual(resp.status_code, 422)
        self.assertIn("%pdf header", resp.json()["detail"].lower())

    def test_decompression_bomb_returns_422(self):
        """Images triggering DecompressionBombError return 422, never 500."""
        with patch("PIL.Image.open") as mock_open:
            mock_open.side_effect = Image.DecompressionBombError("Image exceeds limits")
            resp = self.client.post(
                "/upload",
                files={"file": ("bomb.png", b"\x89PNG\r\n\x1a\nfakeimagebytes", "image/png")}
            )
            self.assertEqual(resp.status_code, 422)
            self.assertIn("decompression bomb", resp.json()["detail"].lower())

    def test_scanned_pdf_ocr_fallback_returns_422_when_empty(self):
        """Scanned PDF with no embedded text and no OCR output returns 422."""
        pdf_doc = pypdfium2.PdfDocument.new()
        pdf_doc.new_page(width=200, height=200)
        pdf_bytes_io = io.BytesIO()
        pdf_doc.save(pdf_bytes_io)
        pdf_bytes = pdf_bytes_io.getvalue()

        with patch("backend.run_tesseract_ocr", return_value=""):
            resp = self.client.post(
                "/upload",
                files={"file": ("scanned_blank.pdf", pdf_bytes, "application/pdf")}
            )
            self.assertEqual(resp.status_code, 422)

    def test_scanned_pdf_ocr_fallback_succeeds(self):
        """Scanned PDF with OCR text extracts successfully via fallback."""
        pdf_doc = pypdfium2.PdfDocument.new()
        pdf_doc.new_page(width=200, height=200)
        pdf_bytes_io = io.BytesIO()
        pdf_doc.save(pdf_bytes_io)
        pdf_bytes = pdf_bytes_io.getvalue()

        with patch("backend.run_tesseract_ocr", return_value="FIRST INFORMATION REPORT UNDER SECTION 154 CRPC"):
            resp = self.client.post(
                "/upload",
                files={"file": ("fir_scan.pdf", pdf_bytes, "application/pdf")}
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("SECTION 154 CRPC", data["extracted_text"])
            self.assertFalse(data["truncated"])

if __name__ == "__main__":
    unittest.main()
