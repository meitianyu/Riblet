import importlib,unittest


class PreprocessingTests(unittest.TestCase):
    def test_pair_roles_and_padding_do_not_change_tokenized_input(self):
        module = importlib.import_module('riblet.preprocessing')
        from tokenizers import Tokenizer,models,pre_tokenizers,processors
        tok=Tokenizer(models.WordPiece({'[UNK]':0,'[CLS]':1,'[SEP]':2,'hello':3,',':4,'world':5,'##s':6,'[PAD]':7},unk_token='[UNK]'))
        tok.pre_tokenizer=pre_tokenizers.BertPreTokenizer()
        tok.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',pair='[CLS] $A [SEP] $B:1 [SEP]:1',special_tokens=[('[CLS]',1),('[SEP]',2)])
        got=module.tokenize_bert('mrpc',{'sentence1':'hello,','sentence2':'worlds'},tok,pad_id=7)
        self.assertEqual(got['effective_length'],7)
        self.assertEqual(got['input_ids'][0,:7].tolist(),[1,3,4,2,5,6,2])
        self.assertEqual(set(got),{'input_ids','attention_mask','token_type_ids','effective_length'})
        self.assertEqual(got['token_type_ids'][0,:7].tolist(),[0,0,0,0,1,1,1])
        self.assertEqual(got['input_ids'].shape,(1,128))
        self.assertEqual(int(got['attention_mask'].sum()),7)
        self.assertTrue((got['input_ids'][0,7:]==7).all())
        with self.assertRaises(ValueError):module.tokenize_bert('mnli',{'premise':'x'},tok,pad_id=7)


if __name__=='__main__':unittest.main()
