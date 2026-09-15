import tempfile
import unittest
from pathlib import Path

from literature_keyword_network import (
    build_topic_config,
    extract_candidates,
    extract_custom_keywords,
    load_custom_keywords,
    relevant_context,
)


class CustomKeywordTests(unittest.TestCase):
    def test_loads_phrases_trims_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keywords.txt"
            path.write_text(
                " Climate Change, renewable energy, ,CLIMATE   CHANGE, biodiversity ",
                encoding="utf-8",
            )
            self.assertEqual(
                load_custom_keywords(path),
                ["climate change", "renewable energy", "biodiversity"],
            )

    def test_matches_phrases_case_insensitively_with_word_boundaries(self):
        terms, binary, df, _, _, _ = extract_custom_keywords(
            [
                "Climate Change increases demand for renewable   energy.",
                "Biodiversity matters; climatic is not the word climate.",
            ],
            ["climate change", "renewable energy", "climate", "missing term"],
        )
        self.assertEqual(terms.tolist(), [
            "climate change", "renewable energy", "climate", "missing term"
        ])
        self.assertEqual(binary.tolist(), [[1, 1, 1, 0], [0, 0, 1, 0]])
        self.assertEqual(df.tolist(), [1, 1, 2, 0])

    def test_automatic_extraction_accepts_terms_from_other_fields(self):
        terms, _, _, _, _, _ = extract_candidates(
            [
                "Quantum entanglement enables secure communication between particles.",
                "Quantum entanglement supports distributed communication systems.",
            ],
            top_n=10,
            min_df=2,
            max_df=1.0,
        )
        self.assertIn("quantum entanglement", terms.tolist())

    def test_guided_extraction_uses_user_terms_as_dynamic_topic(self):
        topic = build_topic_config(["renewable energy", "carbon emissions"])
        terms, _, _, _, _, excluded = extract_candidates(
            [
                "Renewable energy storage reduces carbon emissions from electricity generation.",
                "Renewable energy investment supports low-carbon electricity generation.",
            ],
            top_n=20,
            min_df=1,
            max_df=1.0,
            topic=topic,
        )
        self.assertIn("renewable energy", terms.tolist())
        self.assertNotIn("electricity", terms.tolist())
        self.assertIn("non_topic_single_word", excluded["exclusion_reason"].tolist())

    def test_relevant_context_is_driven_by_user_keywords(self):
        text = (
            "This opening sentence discusses unrelated historical background in sufficient detail. "
            "A nearby sentence introduces current policy and infrastructure constraints. "
            "Renewable energy is central to the proposed electricity transition strategy. "
            "The following sentence describes storage infrastructure and market investment. "
            "This final sentence moves to an unrelated topic outside the selected context."
        )
        selected = relevant_context(text, build_topic_config(["renewable energy"]), window=1)
        self.assertNotIn("opening sentence", selected)
        self.assertIn("current policy", selected)
        self.assertIn("Renewable energy", selected)
        self.assertIn("storage infrastructure", selected)
        self.assertNotIn("final sentence", selected)

    def test_guided_context_covers_morphological_variants(self):
        text = "Predictive models provide useful evidence for the proposed computational framework."
        selected = relevant_context(text, build_topic_config(["prediction"]), window=0)
        self.assertEqual(selected, text)


if __name__ == "__main__":
    unittest.main()
