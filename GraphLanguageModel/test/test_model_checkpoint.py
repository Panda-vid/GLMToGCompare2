import filecmp
from pathlib import Path
import shutil
import unittest

from GraphLanguageModel.pipelines.util import ModelCheckpoint

class TestModelCheckpointConversion(unittest.TestCase):
    def setUp(self):
        self.test_checkpoint_path = Path("./test_data/saved_models/flan-t5-base").resolve()
        self.temporary_path = self.test_checkpoint_path.parent / "temp"
        self.checkpoint_location = shutil.copytree(self.test_checkpoint_path, self.temporary_path)

    def testCheckpointConversion(self):
        modelCheckpoint = ModelCheckpoint(self.checkpoint_location)
        self.assertFalse(all(child.name in ["encoder", "generator", "progress"] for child in self.checkpoint_location.iterdir()))
        test_best_dircmp = filecmp.dircmp(self.test_checkpoint_path, self.checkpoint_location / "best")
        test_latest_dircmp = filecmp.dircmp(self.test_checkpoint_path, self.checkpoint_location / "latest")
        self.assertListEqual(test_best_dircmp.left_list, test_best_dircmp.common)
        self.assertListEqual(test_latest_dircmp.left_list, test_latest_dircmp.common)

    def tearDown(self):
        shutil.rmtree(self.temporary_path)