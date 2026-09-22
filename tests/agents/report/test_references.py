"""보고서 REFERENCE 분류·서지 형식 테스트."""

import unittest

from agents.report.references import collect_references, format_reference


class ReferenceFormattingTests(unittest.TestCase):
    def test_formats_patent_paper_and_other(self):
        patent = {
            "author_or_organization": "NVIDIA",
            "published_date": "2025-04-01",
            "title": "KV Cache Transform Coding",
            "source_type": "patent",
            "locator": "US-XXXXXXX-A1",
            "url": "https://patents.example/US-XXXXXXX-A1",
        }
        paper = {
            "author_or_organization": "Zandieh, A. et al.",
            "published_date": "2025-04-01",
            "title": "TurboQuant: Online Vector Quantization",
            "source_type": "paper",
            "locator": "p. 1",
            "url": "https://arxiv.org/abs/2504.01234",
        }
        other = {
            "author_or_organization": "Google Research",
            "published_date": "2026-03-30",
            "title": "TurboQuant for KV Cache Compression",
            "source_type": "official_web",
            "locator": "blog",
            "url": "https://research.google/blog/turboquant",
        }

        self.assertEqual(
            format_reference(patent),
            "특허 : NVIDIA(2025). *KV Cache Transform Coding*. US-XXXXXXX-A1. "
            "https://patents.example/US-XXXXXXX-A1.",
        )
        self.assertEqual(
            format_reference(paper),
            "논문 : Zandieh, A. et al.(2025). TurboQuant: Online Vector Quantization. "
            "*arXiv*, 2504.01234.",
        )
        self.assertEqual(
            format_reference(other),
            "기타 : Google Research(2026-03-30). *TurboQuant for KV Cache Compression*. "
            "Google Research Blog. https://research.google/blog/turboquant.",
        )

    def test_groups_patent_paper_other_regardless_of_input_order(self):
        store = {
            "other": {
                "source_type": "news",
                "title": "Other",
                "url": "https://example.com/other",
            },
            "paper": {
                "source_type": "paper",
                "title": "Paper",
                "url": "https://arxiv.org/abs/2504.1",
            },
            "patent": {
                "source_type": "patent",
                "title": "Patent",
                "url": "https://example.com/patent",
            },
        }

        lines, _ = collect_references(["other", "paper", "patent"], store)

        self.assertEqual(
            [line.split(" : ", 1)[0] for line in lines],
            ["특허", "논문", "기타"],
        )


if __name__ == "__main__":
    unittest.main()
