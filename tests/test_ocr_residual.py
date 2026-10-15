import copy
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import BertTokenizerFast
from test_data_model import tiny_text
from mact.common.data import Example
from mact.data import SequenceDataset
from mact.model import SequenceModel, ModelConfig
from mact.runtime import save_checkpoint, model_inputs
from mact.ocr_residual import (OCRResidualModel, OCRDataset, OCRCollator,
    residual_train_epoch, save_residual, load_residual)
from mact.ocr_residual_train import shuffled_ocr, OCRIntervention, protected_paths


class OCRResidualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        vocab = self.root/'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\ngood\nwash\n##ington\nbad\n')
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab))
        self.examples = [Example('a', ['good','washington'], 'a.jpg', [(1,2,2)]),
                         Example('b', ['bad'], 'b.jpg', [])]
        self.base = SequenceModel(tiny_text(), ModelConfig(hidden_size=8, attention_heads=2,
                                  visual_size=8, dropout=.5, visual_dropout=1.)).eval()
        self.original = SequenceDataset(self.examples, self.tokenizer)
        self.ds = OCRDataset(self.original, self.tokenizer, {'a.jpg':'good washington bad','b.jpg':''}, max_tokens=2)
        self.collate = OCRCollator(self.tokenizer.pad_token_id)

    def batch(self, empty=False):
        items = [self.ds[i] for i in range(len(self.ds))]
        for item in items:
            item['visual_features'] = torch.randn(3,8)
            if empty:
                item['ocr_content_mask'].zero_()
        return self.collate(items)

    def test_original_alignment_and_independent_budget(self):
        for i in range(2):
            for key in ('input_ids','attention_mask','word_ids','tags'):
                self.assertTrue(torch.equal(self.ds[i][key], self.original[i][key]))
        self.assertEqual(self.ds[0]['ocr_content_mask'].sum().item(),2)
        self.assertEqual(len(self.ds[0]['ocr_input_ids']),4)
        self.assertEqual(self.ds[1]['ocr_content_mask'].sum().item(),0)
        with self.assertRaises(ValueError):
            OCRDataset(self.original,self.tokenizer,{'a.jpg':'hi'})

    def test_zero_init_eval_mode_gradients_and_no_ocr_after_updates(self):
        inputs = model_inputs(self.batch(), 'cpu', include_tags=True)
        original_inputs = {k:v for k,v in inputs.items() if not k.startswith('ocr_')}
        self.base.requires_grad_(False)
        with torch.no_grad():
            expected = self.base(**original_inputs)['emissions']
        before = copy.deepcopy(self.base.state_dict())
        model = OCRResidualModel(self.base, heads=2).train()
        self.assertFalse(model.baseline.training)
        self.assertTrue(all(not m.training for m in model.baseline.modules()))
        initial = model(**inputs)['emissions']
        self.assertTrue(torch.equal(initial, expected))
        optimizer = torch.optim.AdamW(model.branch.parameters(),lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            output = model(**inputs)
            output['losses'].mean().backward()
            optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.baseline.parameters()))
        self.assertGreater(model.branch.attention.in_proj_weight.grad.abs().sum().item(),0)
        for key,value in before.items():
            self.assertTrue(torch.equal(value,model.baseline.state_dict()[key]),key)
        actual = model(**inputs)['emissions']
        self.assertTrue(torch.equal(actual[1],expected[1]))
        self.assertFalse(torch.equal(actual[0],expected[0]))
        inputs['ocr_content_mask'].zero_()
        self.assertTrue(torch.equal(model(**inputs)['emissions'], expected))

    def test_empty_windows_do_not_update_and_mixed_windows_train(self):
        model = OCRResidualModel(self.base,heads=2)
        optimizer = torch.optim.AdamW(model.branch.parameters(),lr=.01)
        before = copy.deepcopy(model.state_dict())
        result = residual_train_epoch(model,[self.batch(empty=True)],optimizer,'cpu',accumulation=1)
        self.assertEqual(result['updates'],0)
        self.assertEqual(result['examples'],2)
        for key,value in before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]))
        result = residual_train_epoch(model,[self.batch()],optimizer,'cpu',accumulation=1)
        self.assertEqual(result['updates'],1)

    def test_checkpoint_roundtrip_and_baseline_hash_guard(self):
        base_path = self.root/'base.pt'
        save_checkpoint(base_path,self.base,{},1,{})
        model = OCRResidualModel(self.base,heads=2).eval()
        with torch.no_grad():
            model.branch.output.weight.normal_()
        inputs = model_inputs(self.batch(),'cpu')
        path = self.root/'branch.pt'
        save_residual(path,model,base_path,{},2,{})
        restored,meta = load_residual(path)
        self.assertEqual(meta['epoch'],2)
        self.assertTrue(torch.equal(model(**inputs)['emissions'], restored(**inputs)['emissions']))
        base_path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'baseline.*hash'):
            load_residual(path)

    def test_nonbaseline_variant_rejected(self):
        self.base.config.word_context=True
        with self.assertRaises(ValueError):
            OCRResidualModel(self.base)

    def test_shuffle_is_bijective_different_content_and_keeps_original_inputs(self):
        examples=[Example(str(i),['good'],str(i)+'.jpg',[]) for i in range(5)]
        original=SequenceDataset(examples,self.tokenizer)
        texts=dict(zip([e.image_id for e in examples],['good','bad','good bad','bad good','']))
        dataset=OCRDataset(original,self.tokenizer,texts)
        donors,details=shuffled_ocr(dataset,[e.image_id for e in examples],17)
        self.assertEqual(set(donors),{0,1,2,3})
        self.assertEqual(set(donors.values()),{0,1,2,3})
        self.assertEqual(details['max_length_difference'],0)
        treatment=OCRIntervention(dataset,donors=donors)
        for i in range(5):
            for key in ('input_ids','attention_mask','word_ids','tags'):
                self.assertTrue(torch.equal(treatment[i][key],original[i][key]))
            if i<4:
                self.assertNotEqual(i,donors[i])
                self.assertFalse(torch.equal(treatment[i]['ocr_input_ids'],dataset[i]['ocr_input_ids']))
            else:
                self.assertFalse(treatment[i]['ocr_content_mask'].any())

    def test_accumulation_includes_no_ocr_constants_in_sentence_mean(self):
        model=OCRResidualModel(self.base,heads=2)
        other=copy.deepcopy(model)
        empty=self.batch(empty=True)
        active=self.batch()
        optimizer=torch.optim.SGD(model.branch.parameters(),lr=.1)
        reference=torch.optim.SGD(other.branch.parameters(),lr=.1)
        other.train()
        losses=other(**model_inputs(active,'cpu',include_tags=True))['losses']
        (losses.sum()/4).backward()
        reference.step()
        residual_train_epoch(model,[empty,active],optimizer,'cpu',accumulation=2,gradient_clip=1e6)
        for name,parameter in model.branch.state_dict().items():
            torch.testing.assert_close(parameter,other.branch.state_dict()[name],rtol=0,atol=0)

    def test_tokenizer_and_feature_payload_are_integrity_protected_without_test(self):
        reference=self.root/'reference'
        (reference/'tokenizer').mkdir(parents=True)
        names=['tokenizer.json','tokenizer_config.json','spm.model','special_tokens_map.json']
        for name in names:
            (reference/'tokenizer'/name).write_text('tokenizer content')
        cfg=dict(baseline=str(self.root/'base.pt'),ocr_cache=str(self.root/'ocr'),
            baseline_run=dict(visual_cache=str(self.root/'visual'),train_data=str(self.root/'train.json'),
                              dev_data=str(self.root/'dev.json')))
        paths=protected_paths(cfg,self.root/'config.json',reference)
        for name in names:self.assertIn(reference/'tokenizer'/name,paths)
        self.assertIn(self.root/'visual/features.pt',paths)
        self.assertNotIn(self.root/'test.json',paths)


if __name__=='__main__': unittest.main()
