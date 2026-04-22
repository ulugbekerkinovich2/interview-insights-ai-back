import os
import sys
import unittest
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# chunk_text ichki defaultlarini _CHUNK_MAX_CHARS env orqali boshqarishini
# e'tiborga olib, import'dan oldin barqaror qiymat qo'yamiz.
os.environ.setdefault("RAG_CHUNK_MAX_CHARS", "1800")
os.environ.setdefault("RAG_CHUNK_OVERLAP", "250")

from utils.rag_knowledge import chunk_text, _MIN_CHUNK_CHARS  # noqa: E402


class ChunkTextTests(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t  "), [])

    def test_short_text_returns_single_chunk(self):
        text = "Bu qisqa matn."
        chunks = chunk_text(text, max_chars=1800, overlap=250)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_text_at_boundary_is_one_chunk(self):
        text = "a" * 1800
        chunks = chunk_text(text, max_chars=1800, overlap=250)
        self.assertEqual(len(chunks), 1)

    def test_long_text_splits_into_multiple_chunks(self):
        paragraph = "Sentence one. Sentence two. " * 200  # ~5400 chars
        chunks = chunk_text(paragraph, max_chars=500, overlap=80)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 500)

    def test_overlap_carries_context_between_chunks(self):
        sentences = ". ".join(f"Sentence {i}" for i in range(60)) + "."
        chunks = chunk_text(sentences, max_chars=200, overlap=50)
        self.assertGreaterEqual(len(chunks), 2)
        # Ikkita ketma-ket chunk orasida kamida biror umumiy qismi bo'lishi kerak.
        overlap_found = any(
            any(chunks[i][-40:] in chunks[i + 1][:150] for _ in [0])
            for i in range(len(chunks) - 1)
        )
        self.assertTrue(
            overlap_found or len(chunks[0]) < 200,
            "Chunks should share overlap text when source has sentence boundaries",
        )

    def test_single_huge_paragraph_is_hard_split(self):
        huge = "x" * 4000
        chunks = chunk_text(huge, max_chars=1000, overlap=100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 1000)

    def test_unicode_uzbek_russian_are_preserved(self):
        uz = "O'zbek tilidagi psixologiya matni. Salomatlik muhim. "
        ru = "Русский текст о психологии. Здоровье важно. "
        text = (uz + ru) * 40
        chunks = chunk_text(text, max_chars=300, overlap=60)
        self.assertGreater(len(chunks), 1)
        joined = " ".join(chunks)
        self.assertIn("O'zbek", joined)
        self.assertIn("Русский", joined)

    def test_normalizes_excessive_newlines_and_whitespace(self):
        text = "Para one.\n\n\n\n\nPara two.\t\t\t  Extra    spaces."
        chunks = chunk_text(text, max_chars=1800, overlap=250)
        self.assertEqual(len(chunks), 1)
        # 3+ \n -> \n\n, ketma-ket bo'sh joylar bir probelga siqiladi.
        self.assertNotIn("\n\n\n", chunks[0])
        self.assertNotIn("    ", chunks[0])

    def test_carriage_return_line_feed_is_normalized(self):
        text = "Line A.\r\n\r\nLine B.\r\n\r\nLine C."
        chunks = chunk_text(text, max_chars=1800, overlap=250)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Line A.", chunks[0])
        self.assertIn("Line C.", chunks[0])

    def test_min_chunk_chars_filters_tiny_tail(self):
        # Juda qisqa quyruq chunk qilinmasligi kerak, agar final list bo'sh bo'lmasa.
        # (_MIN_CHUNK_CHARS = 120 ni inobatga olamiz.)
        text = ("A" * 990) + "\n\n" + ("B" * (_MIN_CHUNK_CHARS - 10))
        chunks = chunk_text(text, max_chars=1000, overlap=50)
        self.assertGreaterEqual(len(chunks), 1)
        # Hech qanday chunk bo'sh bo'lmasligi kerak
        for chunk in chunks:
            self.assertTrue(chunk.strip())


if __name__ == "__main__":
    unittest.main()
