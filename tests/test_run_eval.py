import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from run_eval import (
    brief_path,
    is_valid_trajectory,
    load_benchmark,
    safe_api_url,
    select_benchmarks,
    trajectory_path,
    validate_name,
    write_brief,
)


FIXTURES = Path(__file__).parent / "fixtures"


class RunEvalTests(unittest.TestCase):
    def test_load_json_and_jsonl_with_custom_fields(self):
        spec = {"problem_field": "q", "answer_field": "a", "id_field": "uid"}
        self.assertEqual(load_benchmark(FIXTURES / "benchmark.json", spec), [
            {"example_id": "x", "problem": "P", "answer": "A"}
        ])
        self.assertEqual(load_benchmark(FIXTURES / "benchmark.jsonl", {}), [
            {"example_id": 0, "problem": "P2", "answer": "A2"}
        ])

    def test_select_benchmarks(self):
        config = {"benchmarks": {"a": {}, "b": {}}}
        self.assertEqual(select_benchmarks(config, "all"), ["a", "b"])
        self.assertEqual(select_benchmarks(config, "b,a"), ["b", "a"])
        with self.assertRaises(ValueError):
            select_benchmarks(config, "missing")

    def test_output_paths_and_name_validation(self):
        root = Path("eval_results") / "cmp" / "model_a"
        self.assertEqual(
            trajectory_path(root, "aime", 2, 3),
            root / "aime" / "run_2" / "traj_3.json",
        )
        self.assertEqual(brief_path(root, "aime", 2), root / "aime" / "run_2" / "brief.json")
        self.assertEqual(validate_name("model-a.1", "model"), "model-a.1")
        with self.assertRaises(ValueError):
            validate_name("../model", "model")

    def test_trajectory_validation(self):
        self.assertTrue(is_valid_trajectory(FIXTURES / "valid_traj.json"))
        self.assertFalse(is_valid_trajectory(FIXTURES / "invalid_traj.json"))

    def test_write_brief_summary_without_filesystem_write(self):
        score_file = io.StringIO('{"score": 1.0}')
        with patch("run_eval.is_valid_trajectory", return_value=True), \
             patch("pathlib.Path.open", return_value=score_file), \
             patch("run_eval.write_json") as write_json_mock:
            self.assertTrue(write_brief(Path("root"), "bench", 1, 1, 1.0))
        payload = write_json_mock.call_args.args[1]
        self.assertEqual(payload["resolved"], 1)
        self.assertEqual(payload["score"], 1.0)

    def test_safe_api_url_removes_credentials_and_query(self):
        self.assertEqual(
            safe_api_url("https://user:secret@example.com:8443/v1?token=x#part"),
            "https://example.com:8443/v1",
        )


if __name__ == "__main__":
    unittest.main()
