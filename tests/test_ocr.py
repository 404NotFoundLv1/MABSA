import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

import torch
from mact.data import SequenceDataset, SequenceCollator
from mact.model import SequenceModel, ModelConfig
from mact.runtime import model_inputs
from mact.train import resolve_config
from mact.common.cache import file_hash
import test_data_model as fixtures


class OCRTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DataModelTests(); self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.tok = self.fixture.tokenizer
        self.examples = self.fixture.examples

    def test_suffix_preserves_original_words_and_budget(self):
        baseline = SequenceDataset(self.examples, self.tok, max_length=5)
        ds = SequenceDataset(self.examples, self.tok, max_length=5,
            ocr_texts={'a.jpg':'bad good washington', 'b.jpg':''}, ocr_max_tokens=2)
        a, b = ds[0], baseline[0]
        self.assertEqual(a['input_ids'][:len(b['input_ids'])].tolist(), b['input_ids'].tolist())
        self.assertEqual(a['input_ids'][len(b['input_ids']):].tolist(), [8, 5, self.tok.sep_token_id])
        self.assertEqual(a['word_ids'].tolist(), b['word_ids'].tolist()+[-1]*3)
        self.assertTrue(torch.equal(a['tags'], b['tags']))
        batch = SequenceCollator(self.tok.pad_token_id)([ds[0],ds[1]])
        self.assertEqual(batch['word_mask'].sum().item(), 3)
        for key in ('input_ids','word_ids','tags','attention_mask'):
            self.assertTrue(torch.equal(ds[1][key],baseline[1][key]))
        with self.assertRaises(ValueError):
            SequenceDataset(self.examples,self.tok,max_length=4,ocr_texts={'a.jpg':'','b.jpg':''})

    def test_missing_ocr_fails_and_gold_does_not_change_encoding(self):
        with self.assertRaises(ValueError):
            SequenceDataset(self.examples,self.tok,ocr_texts={})
        rows=[self.examples[0],replace(self.examples[0],id='other',aspects=[])]
        ds=SequenceDataset(rows,self.tok,ocr_texts={'a.jpg':'bad'})
        for key in ('input_ids','attention_mask','word_ids'):
            self.assertTrue(torch.equal(ds[0][key],ds[1][key]))

    def test_context_receives_gradient_but_never_decode_positions(self):
        rows=[replace(self.examples[0],words=['good'],aspects=[(0,1,2)])]
        ds=SequenceDataset(rows,self.tok,ocr_texts={'a.jpg':'bad'})
        b=SequenceCollator(self.tok.pad_token_id)([ds[0]])
        model=SequenceModel(fixtures.tiny_text(),ModelConfig(mode='text',hidden_size=8,attention_heads=2))
        out=model(**model_inputs(b,'cpu',True));out['losses'].sum().backward()
        self.assertGreater(model.text_encoder.embeddings.word_embeddings.weight.grad[8].abs().sum().item(),0)
        self.assertEqual(len(model.crf.decode(out['emissions'],b['word_mask'])[0]),1)

    def test_asset_rejects_missing_changed_image_and_changed_payload(self):
        from mact.ocr import open_ocr
        root=Path(self.fixture.tmp.name);cache=root/'ocr';cache.mkdir()
        (root/'a.jpg').write_bytes(b'image')
        data={'a.jpg':{'image_sha256':file_hash(root/'a.jpg'),'text':'bad'}}
        (cache/'texts.json').write_text(json.dumps(data))
        manifest={'format':'mact_ocr_v1','texts_sha256':file_hash(cache/'texts.json')}
        (cache/'manifest.json').write_text(json.dumps(manifest))
        expected=file_hash(cache/'manifest.json')
        texts,_=open_ocr(cache,root,self.examples[:1],expected)
        self.assertEqual(texts,{'a.jpg':'bad'})
        with self.assertRaises(ValueError):open_ocr(cache,root,self.examples,expected)
        (root/'a.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError):open_ocr(cache,root,self.examples[:1],expected)
        (cache/'texts.json').write_text('{}')
        with self.assertRaises(ValueError):open_ocr(cache,root,self.examples[:1],expected)

    def test_generic_filter_and_configuration_guards(self):
        from mact.ocr import filter_lines
        self.assertEqual(filter_lines([
            {'text':'  good   day ', 'confidence':.9},
            {'text':'bad','confidence':.79},{'text':'0000','confidence':.99},
            {'text':'图','confidence':.95},{'text':'ABC','confidence':.8}]),'good day ABC')
        self.assertIsNone(resolve_config({})['ocr_cache'])
        for raw in ({'ocr_cache':'x','rdrop_weight':1},
                    {'ocr_cache':'x','model':{'word_context':True}},
                    {'ocr_cache':'x','ocr_max_tokens':0},
                    {'ocr_cache':'x','max_length':512}):
            with self.assertRaises(ValueError):resolve_config(raw)

    def test_train_only_reloads_ocr_and_does_not_require_test_file(self):
        import mact.experiment as experiment
        root=Path(self.fixture.tmp.name)
        model=root/'model';fixtures.tiny_text().save_pretrained(model);self.tok.save_pretrained(model)
        vision=root/'vision';vision.mkdir();(vision/'config.json').write_text('{}')
        images=root/'images';images.mkdir();(images/'a.jpg').write_bytes(b'image')
        from mact.common.cache import model_hashes
        cache=root/'cache';cache.mkdir();torch.save({'a.jpg':torch.ones(2,8)},cache/'features.pt')
        (cache/'manifest.json').write_text(json.dumps(dict(format_version=1,vision_model='../vision',image_root='../images',
            model_sha256=model_hashes(vision),features_sha256=file_hash(cache/'features.pt'),
            image_sha256={'a.jpg':file_hash(images/'a.jpg')},token_shape=[2,8])))
        ocr=root/'ocr';ocr.mkdir()
        (ocr/'texts.json').write_text(json.dumps({'a.jpg':dict(image_sha256=file_hash(images/'a.jpg'),text='bad')}))
        (ocr/'manifest.json').write_text(json.dumps(dict(format='mact_ocr_v1',texts_sha256=file_hash(ocr/'texts.json'))))
        for split in ('train','dev'):
            (root/f'{split}.json').write_text(json.dumps([dict(id=split,words=['good','washington'],image_id='a.jpg',
                aspects=[{'from':1,'to':2,'polarity':'positive'}])]))
        (root/'configs/datasets').mkdir(parents=True)
        (root/'configs/base.json').write_text(json.dumps(dict(epochs=1,log_every=0,device='cpu',num_threads=1,
            text_model='model',vision_model='vision',ocr_cache='ocr',
            model=dict(mode='fusion',hidden_size=8,attention_heads=2,visual_size=8))))
        (root/'configs/datasets/twitter2015.json').write_text(json.dumps(dict(train_data='train.json',dev_data='dev.json',
            test_data='FORBIDDEN-NONEXISTENT.json',image_root='images',visual_cache='cache',test_visual_cache='NO-TEST-CACHE')))
        with patch.object(experiment,'ROOT',root):
            out=root/'outputs/run'
            summary=experiment.run_experiment('configs/base.json','twitter2015',seed=42,output=out,evaluate_test=False)
        self.assertEqual(summary['completed_epochs'],1)
        self.assertTrue(json.loads((out/'reload-verification.json').read_text())['predictions_exact'])
        self.assertEqual(json.loads((out/'config.json').read_text())['ocr_manifest_sha256'],file_hash(ocr/'manifest.json'))
        self.assertFalse((out/'test').exists())
        self.assertEqual(len(json.loads((out/'experiment.json').read_text())['data_sha256']),2)
        from mact.evaluate import run_evaluation
        (ocr/'manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            run_evaluation(out/'best.pt',root/'dev.json',root/'corrupt-reload')

if __name__=='__main__':unittest.main()
