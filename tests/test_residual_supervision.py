"""Catch loss saturation, OCR-content leakage into controls and incompatible reloads."""
import copy
import json
import unittest
import torch
import test_ocr_residual as fixture
from mact.ocr_residual import OCRResidualModel, save_residual, load_residual, residual_train_epoch
from mact.runtime import model_inputs, save_checkpoint
from mact.ocr_residual_train import read_config
from mact.ocr_residual_results import same_treatment
from mact.experiment import ROOT


class ResidualSupervisionTests(unittest.TestCase):
    setUp=fixture.OCRResidualTests.setUp
    batch=fixture.OCRResidualTests.batch

    def test_direct_loss_supplies_gradient_when_fused_loss_is_saturated(self):
        with torch.no_grad():
            self.base.text_head.weight.zero_();self.base.text_head.bias.zero_();self.base.text_head.bias[0]=80
            self.base.visual_head[-1].weight.zero_();self.base.visual_head[-1].bias.zero_()
        inputs=model_inputs(self.batch(),'cpu',include_tags=True);inputs['tags'].zero_()
        model=OCRResidualModel(self.base,heads=2,standalone_weight=1.)
        output=model(**inputs)
        joint_grad=torch.autograd.grad(output['joint_losses'].sum(),model.branch.output.weight,retain_graph=True)[0]
        total_grad=torch.autograd.grad(output['losses'].sum(),model.branch.output.weight)[0]
        self.assertLess(float(joint_grad.norm()),1e-12)
        self.assertGreater(float(total_grad.norm()),.001)
        self.assertEqual(float(output['standalone_losses'][1]),0.)
        self.assertGreater(float(output['standalone_losses'][0]),1.)
        torch.testing.assert_close(output['losses'],output['joint_losses']+output['standalone_losses'])
        self.assertTrue(torch.equal(output['emissions'],output['base_emissions']))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.base.parameters()))

    def test_text_control_ignores_ocr_content_with_identical_capacity_and_bypass(self):
        inputs=model_inputs(self.batch(),'cpu',include_tags=True)
        models=[]
        for source in ('ocr','text'):
            torch.manual_seed(17)
            models.append(OCRResidualModel(copy.deepcopy(self.base),heads=2,memory_source=source,standalone_weight=1.).eval())
        self.assertEqual(sum(p.numel() for p in models[0].branch.parameters()),sum(p.numel() for p in models[1].branch.parameters()))
        for a,b in zip(models[0].branch.parameters(),models[1].branch.parameters()):self.assertTrue(torch.equal(a,b))
        for model in models:
            with torch.no_grad():model.branch.output.weight.fill_(.2);model.branch.output.weight[0].fill_(-.4)
        changed={k:v.clone() for k,v in inputs.items()}
        changed['ocr_input_ids'][changed['ocr_content_mask']]=8
        for model in models:
            a,b=model(**inputs),model(**changed)
            if model.memory_source=='text':
                self.assertTrue(torch.equal(a['emissions'],b['emissions']))
                self.assertTrue(torch.equal(a['standalone_losses'],b['standalone_losses']))
            else:self.assertFalse(torch.equal(a['emissions'],b['emissions']))
            self.assertTrue(torch.equal(a['emissions'][1],a['base_emissions'][1]))
            empty={k:v.clone() for k,v in inputs.items()};empty['ocr_content_mask'].zero_()
            removed=model(**empty)
            self.assertTrue(torch.equal(removed['emissions'],removed['base_emissions']))
            self.assertTrue(torch.equal(removed['standalone_losses'],torch.zeros(2)))

    def test_supervised_updates_leave_baseline_and_empty_windows_unchanged(self):
        model=OCRResidualModel(self.base,heads=2,memory_source='text',standalone_weight=1.)
        before=copy.deepcopy(model.baseline.state_dict());branch=copy.deepcopy(model.branch.state_dict())
        optimizer=torch.optim.AdamW(model.branch.parameters(),lr=.01)
        result=residual_train_epoch(model,[self.batch(empty=True)],optimizer,'cpu',accumulation=1)
        self.assertEqual(result['updates'],0);self.assertEqual(result['standalone_nll'],0.)
        for name,p in model.branch.state_dict().items():self.assertTrue(torch.equal(p,branch[name]))
        result=residual_train_epoch(model,[self.batch(),self.batch()],optimizer,'cpu',accumulation=1)
        self.assertEqual(result['updates'],2)
        self.assertAlmostEqual(result['objective_loss'],result['sentence_nll']+result['standalone_nll'],places=5)
        self.assertGreater(float(model.branch.attention.in_proj_weight.grad.norm()),0.)
        for name,p in model.baseline.state_dict().items():self.assertTrue(torch.equal(p,before[name]))
        self.assertTrue(all(not m.training for m in model.baseline.modules()))

    def test_zero_aux_weight_preserves_old_outputs_and_training_loss(self):
        inputs=model_inputs(self.batch(),'cpu',include_tags=True)
        torch.manual_seed(19);old=OCRResidualModel(copy.deepcopy(self.base),heads=2)
        torch.manual_seed(19);new=OCRResidualModel(copy.deepcopy(self.base),heads=2,memory_source='ocr',standalone_weight=0.)
        a,b=old(**inputs),new(**inputs)
        self.assertTrue(torch.equal(a['losses'],b['losses']))
        self.assertTrue(torch.equal(a['emissions'],b['emissions']))

    def test_text_memory_excludes_special_and_padded_tokens_and_obeys_budget(self):
        inputs=model_inputs(self.batch(),'cpu',include_tags=True)
        model=OCRResidualModel(self.base,heads=2,memory_source='text',standalone_weight=1.,memory_max_tokens=2)
        seen=[]
        hook=model.branch.register_forward_pre_hook(lambda m,a:seen.append(a[2].clone()))
        out=model(**inputs);hook.remove()
        expected=(inputs['word_ids'][0]>=0)&inputs['attention_mask'][0].bool()
        positions=expected.nonzero().flatten().tolist()
        expected.zero_();expected[positions[:2]]=True
        self.assertTrue(torch.equal(seen[0][0],expected))
        self.assertTrue(torch.isfinite(out['losses']).all())

    def test_checkpoint_retains_arm_and_loss_and_accepts_legacy_format(self):
        path=self.root/'base.pt';save_checkpoint(path,self.base,{},1,{})
        inputs=model_inputs(self.batch(),'cpu',include_tags=True)
        for source in ('ocr','text'):
            model=OCRResidualModel(copy.deepcopy(self.base),heads=2,memory_source=source,standalone_weight=1.,memory_max_tokens=2).eval()
            with torch.no_grad():model.branch.output.weight.normal_()
            checkpoint=self.root/(source+'.pt');save_residual(checkpoint,model,path,{},1,{})
            restored,_=load_residual(checkpoint)
            self.assertEqual(restored.memory_source,source);self.assertEqual(restored.standalone_weight,1.)
            for key in ('emissions','losses','standalone_losses'):
                torch.testing.assert_close(model(**inputs)[key],restored(**inputs)[key],rtol=0,atol=0)
        old=OCRResidualModel(copy.deepcopy(self.base),heads=2).eval();checkpoint=self.root/'old.pt'
        save_residual(checkpoint,old,path,{},1,{})
        data=torch.load(checkpoint,weights_only=True)
        for key in ('memory_source','standalone_weight','memory_max_tokens'):data.pop(key,None)
        torch.save(data,checkpoint);restored,_=load_residual(checkpoint)
        self.assertEqual(restored.memory_source,'ocr');self.assertEqual(restored.standalone_weight,0.)
        self.assertTrue(torch.equal(old(**inputs)['emissions'],restored(**inputs)['emissions']))

    def test_config_and_aggregation_reject_invalid_or_mixed_arms(self):
        baseline=json.loads((ROOT/'configs/ocr_residual.json').read_text())
        path=self.root/'config.json'
        for update in ({'memory_source':'image'},{'standalone_weight':-1},{'standalone_weight':float('nan')},{'standalone_weight':True}):
            path.write_text(json.dumps(dict(baseline,**update)))
            with self.assertRaises(ValueError):read_config(path,'twitter2015','cpu')
        path.write_text(json.dumps(dict(baseline,memory_source='text',standalone_weight=1.)))
        cfg=read_config(path,'twitter2015','cpu');self.assertEqual(cfg['memory_source'],'text')
        old=read_config(ROOT/'configs/ocr_residual.json','twitter2015','cpu')
        default=dict(old,memory_source='ocr',standalone_weight=0.)
        self.assertTrue(same_treatment(old,default))
        self.assertFalse(same_treatment(old,cfg))
        self.assertFalse(same_treatment(cfg,dict(cfg,memory_source='ocr')))
        self.assertFalse(same_treatment(cfg,dict(cfg,standalone_weight=.1)))

if __name__=='__main__':unittest.main()
