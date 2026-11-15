import importlib
from pathlib import Path
import tempfile,unittest


class AssetIdentityTests(unittest.TestCase):
    def test_same_architecture_does_not_authorize_other_checkpoint_bytes(self):
        module = importlib.import_module('riblet.assets')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'config.json').write_text('{"model_type":"bert"}')
            (root/'flax_model.msgpack').write_bytes(b'unmatched weights')
            with self.assertRaises(ValueError):module.verify_model('bert-cola',root)
            with self.assertRaises(ValueError):module.verify_model('unregistered-model',root)


if __name__=='__main__':unittest.main()
