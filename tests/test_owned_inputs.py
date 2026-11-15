import importlib,tempfile,unittest
from pathlib import Path


class OwnedInputsTests(unittest.TestCase):
    def test_model_party_cannot_open_client_input_or_receive_plain_output(self):
        m = importlib.import_module('riblet.inputs')
        with tempfile.TemporaryDirectory() as temp:
            missing=Path(temp)/'private-input-that-does-not-exist.npy'
            with self.assertRaises(PermissionError):m.load_input(0,'bert-cola',missing,Path(temp))
            with self.assertRaises(PermissionError):m.save_output(0,'bert-cola',[1,2],Path(temp)/'out')

    def test_task_output_shape_and_regression_are_not_binary_guesses(self):
        m = importlib.import_module('riblet.inputs')
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            r=m.save_output(1,'bert-stsb',np.array([[2.75]],np.float32),Path(temp)/'regression')
            self.assertEqual(r['output_kind'],'regression');self.assertEqual(r['value'],2.75)
            r=m.save_output(1,'bert-mnli',np.array([[0.,1.,-1.]],np.float32),Path(temp)/'classification')
            self.assertEqual(r['prediction'],1)
            with self.assertRaises(ValueError):m.save_output(1,'bert-mnli',np.zeros((1,2)),Path(temp)/'bad')


if __name__=='__main__':unittest.main()
