from pathlib import Path
import copy
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "audit"))
from isolation import inventory
from runtime_policy import verify_policy


class RuntimePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(__file__).absolute().parents[1] / ".runtime/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temporary.name)
        (self.root / "app").mkdir()
        (self.root / "app/legacy.py").write_bytes(b"original")
        self.baseline = inventory(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_runtime_activity_is_reported_without_rewriting_source_baseline(self) -> None:
        original = (self.root / "app/legacy.py").read_bytes()
        output = self.root / "app/output"
        output.mkdir()
        (output / "render.mp4").write_bytes(b"runtime data")
        result = verify_policy(self.root, self.baseline)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["runtime_activity"]["observed_files"], 1)
        self.assertEqual((self.root / "app/legacy.py").read_bytes(), original)

    def test_runtime_policy_allows_unified_source_changes(self) -> None:
        (self.root / "app/legacy.py").write_bytes(b"modified")
        self.assertTrue(verify_policy(self.root, self.baseline)["ok"])

    def test_runtime_prefix_does_not_misclassify_local_source(self) -> None:
        (self.root / "app/output_evil").mkdir()
        (self.root / "app/output_evil/code.py").write_bytes(b"forbidden")
        result = verify_policy(self.root, self.baseline)
        self.assertTrue(result["ok"])
        self.assertEqual(result["runtime_activity"]["observed_files"], 0)

    def test_runtime_config_is_reported_and_local_defaults_remain_editable(self) -> None:
        (self.root / "app/run_history.json").write_bytes(b"runtime")
        config = self.root / "app/config.yaml"
        config.write_bytes(b"runtime config v1")
        result = verify_policy(self.root, self.baseline)
        self.assertTrue(result["ok"], result)
        self.assertEqual({item["path"] for item in result["runtime_activity"]["files"]},
                         {"app/config.yaml", "app/run_history.json"})

        default_config = self.root / "app/default_config.yaml"
        default_config.write_bytes(b"protected defaults")
        baseline = inventory(self.root)
        config.write_bytes(b"runtime config v2")
        default_config.write_bytes(b"modified defaults")
        self.assertTrue(verify_policy(self.root, baseline)["ok"])

    def test_queue_activity_is_reported_without_rewriting_baseline(self) -> None:
        queue = self.root / "app/queue.json"
        queue.write_bytes(b'{"jobs": []}')
        baseline = inventory(self.root)
        preserved = copy.deepcopy(baseline)
        queue.write_bytes(b'{"jobs": [{"status": "running"}]}')
        result = verify_policy(self.root, baseline)
        self.assertTrue(result["ok"], result)
        self.assertEqual(baseline, preserved)
        self.assertEqual([entry["path"] for entry in result["runtime_activity"]["files"]], ["app/queue.json"])
        (self.root / "app/queue.json.bak").write_bytes(b"unclassified")
        self.assertTrue(verify_policy(self.root, baseline)["ok"])

    def test_profile_activity_does_not_claim_model_weights_are_runtime_json(self) -> None:
        profiles = self.root / "app/models/runtime_profiles"
        profiles.mkdir(parents=True)
        # Runtime classification is separate from model byte verification in ModelRegistry.
        baseline = inventory(self.root)
        (profiles / "runtime.json").write_bytes(b"{}")
        assert verify_policy(self.root, baseline)["ok"]
        (profiles / "weights.onnx").write_bytes(b"model")
        result = verify_policy(self.root, baseline)
        assert result["ok"]
        assert [item["path"] for item in result["runtime_activity"]["files"]] == ["app/models/runtime_profiles/runtime.json"]


    def test_runtime_link_cannot_escape_the_checkout(self) -> None:
        import os
        from roop_3d.audit.isolation import AuditError, safe_path
        outside = self.root.parent / (self.root.name + '_outside.json')
        outside.write_bytes(b'external data')
        link = self.root / 'app/config.yaml'
        try:
            try:
                os.symlink(outside, link)
            except OSError as error:
                self.skipTest(f'Symlink creation unavailable: {error}')
            result = verify_policy(self.root, self.baseline)
            self.assertFalse(result['ok'])
            self.assertEqual(result['violations'][0]['status'], 'LINK_TRAVERSAL')
            with self.assertRaises(AuditError):
                safe_path(self.root, 'app/config.yaml', write=True)
            self.assertEqual(outside.read_bytes(), b'external data')
        finally:
            outside.unlink(missing_ok=True)
