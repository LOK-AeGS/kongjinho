"""실제 Report LLM 연결 통합 테스트.

저장소의 .env에서 OPENAI_API_KEY를 자동 로드한다. 키가 있으면 실제 API 비용이 발생한다.
"""

import json
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agents.report.state import ReportAgentDeps  # noqa: E402
from agents.report.subgraph import _writer_context, normalize_state  # noqa: E402
from agents.report.validators import blocking, validate_section  # noqa: E402
from agents.report.writer import create_llm_writer, load_report_environment  # noqa: E402


FIXTURE = Path(__file__).with_name("fixtures") / "report_cases.json"
HAS_KEY = load_report_environment()


@unittest.skipUnless(
    HAS_KEY,
    "실제 LLM 테스트는 저장소 .env의 OPENAI_API_KEY가 필요함",
)
class ReportLLMIntegrationTests(unittest.TestCase):
    def test_real_llm_writes_grounded_market_section(self):
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        state = deepcopy(payload["base"])
        context = normalize_state(state)
        deps = ReportAgentDeps(
            generation_mode="llm",
            model=os.getenv("REPORT_TEST_MODEL", "gpt-4o-mini"),
            temperature=0,
        )
        writer = create_llm_writer(deps)
        writer_context = _writer_context(
            "market", context, include_normalized=False
        )

        draft = writer.write("market", writer_context)
        issues = validate_section(draft, context)

        self.assertTrue(draft["markdown"].startswith("## 4.2 시장성"))
        self.assertTrue(set(draft["claim_ids"]).issubset(context["claims"]))
        self.assertTrue(set(draft["evidence_ids"]).issubset(context["evidence_store"]))
        self.assertEqual(blocking(issues), [], issues)


if __name__ == "__main__":
    unittest.main()
