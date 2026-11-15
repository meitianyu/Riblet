import importlib,json,hashlib,unittest


class DevelopmentBuilderTests(unittest.TestCase):
    def test_frozen_row_fold_and_token_content_are_preserved(self):
        m = importlib.import_module('riblet.development')
        from tokenizers import Tokenizer,models,processors,pre_tokenizers
        tok=Tokenizer(models.WordPiece({'[UNK]':0,'[CLS]':1,'[SEP]':2,'hello':3,'[PAD]':4},unk_token='[UNK]'))
        tok.pre_tokenizer=pre_tokenizers.BertPreTokenizer();tok.post_processor=processors.TemplateProcessing(single='[CLS] $A [SEP]',special_tokens=[('[CLS]',1),('[SEP]',2)])
        expected={'schema':'cdi-role6-canonical-input-v1','tokenizer_sha256':'b'*64,'effective_length':3,'input_ids':[1,3,2],'token_type_ids':[0,0,0]}
        member={'task':'cola','source_row_index':0,'example_id':'42','request_digest':'a'*64,'request_ordinal':7,
            'fold_id':3,'source_role':'Development','tokenizer_sha256':'b'*64,
            'content_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        row=m.rebuild_record(member,{'sentence':'hello','idx':42,'label':1},tok)
        self.assertEqual(row['request_ordinal'],7);self.assertEqual(row['fold_id'],3)
        self.assertEqual(row['input_ids'],[1,3,2]);self.assertNotIn('role6_ids',row)
        with self.assertRaises(ValueError):m.rebuild_record(member,{'sentence':'hello hello','idx':42,'label':1},tok)


if __name__=='__main__':unittest.main()
