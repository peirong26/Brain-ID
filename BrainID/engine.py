
"""
Train and eval functions
"""
import os
import sys
import math 

import torch

import utils.misc as utils
import utils.logging as logging


logger = logging.get_logger(__name__)




def make_results(subjects, samples, outputs, out_dir): 
    case_names = subjects['name']
    results = outputs
    case_out_dir = utils.make_dir(os.path.join(out_dir, case_names[0], 'results'))

    if 'aff' in subjects:
        aff = subjects['aff'][0]
    else:
        aff = None

    if 'label' in subjects:
        utils.viewVolume(subjects['label'], aff = aff, names = ['label'], prefix = 'gt_', save_dir = case_out_dir) 
    if 'image' in subjects:
        utils.viewVolume(subjects['image'], aff = aff, names = ['image'], prefix = 'gt_', save_dir = case_out_dir)  
    if 'image_orig' in subjects:
        utils.viewVolume(subjects['image_orig'], aff = aff, names = ['image_orig'], prefix = 'gt_', save_dir = case_out_dir)   
    
    for i_sample, sample in enumerate(samples):

        if 'bias_field_log' in sample:
            utils.viewVolume(torch.exp(sample['bias_field_log']), aff = aff, names = ['bflog'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(torch.exp(outputs[i_sample]['bias_field_log']), aff = aff, names = ['bflog'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

        if 'input' in sample:
            utils.viewVolume(sample['input'], aff = aff, names = ['input'], prefix = '', postfix = '_#%d' % i_sample, save_dir = case_out_dir)
        
        if 'orig' in sample:
            utils.viewVolume(sample['orig'], aff = aff, names = ['orig'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

        if 'source' in sample:
            utils.viewVolume(sample['source'], aff = aff, names = ['source'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(sample['target'], aff = aff, names = ['target'], prefix = 'gt_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(outputs[i_sample]['tgt_def'], aff = aff, names = ['source'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 
            utils.viewVolume(outputs[i_sample]['src_def'], aff = aff, names = ['target'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

        if 'label' in outputs[i_sample]:
            utils.viewVolume(outputs[i_sample]['label'], aff = aff, names = ['label'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir)

        if 'image' in outputs[i_sample]:
            utils.viewVolume(outputs[i_sample]['image'], aff = aff, names = ['image'], prefix = 'pd_', postfix = '_#%d' % i_sample, save_dir = case_out_dir) 

    return results



def train_one_epoch_feature(epoch, args, model, processors, criterion, data_loader, 
                            optimizer, lr_scheduler, wd_scheduler, 
                            postprocessor, visualizers, output_dir, device = 'cpu'):
    
    
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(
        args.log_itr,
        delimiter="  ",
        debug=args.debug)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, args.n_epochs)

    for itr, (subjects, samples) in enumerate(metric_logger.log_every(data_loader, epoch, header=header, dataset_name=args.dataset, modality=args.modality, train_limit=args.train_itr_limit)): 
        
        # update weight decay and learning rate according to their schedule
        itr = len(data_loader) * epoch + itr  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups): 
            param_group["lr"] = lr_scheduler[itr]
            param_group["weight_decay"] = wd_scheduler[itr]

        samples = utils.nested_dict_to_device(samples, device)
        subjects = utils.nested_dict_to_device(subjects, device)

        outputs, _ = model(samples)
        for processor in processors:
            outputs = processor(outputs, samples)
            
        loss_dict = criterion(outputs, subjects, samples) 

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            logger.info(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            utils.clip_gradients(model, args.clip_max_norm)
        utils.cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)
        optimizer.step()
        
        # logging
        if utils.get_world_size() > 1:
            torch.cuda.synchronize()
        metric_logger.update(loss = loss_value,
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled
                            )
        metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.update(wd = optimizer.param_groups[0]["weight_decay"]) 

        if args.debug or (itr % args.vis_itr == 0) and visualizers is not None and utils.is_main_process(): 
            epoch_vis_dir = utils.make_dir(os.path.join(output_dir, str(epoch), str(itr))) if epoch is not None else output_dir

            if postprocessor is not None:
                outputs = postprocessor(args, outputs, task = args.task)
            if args.visualizer.make_results:  
                make_results(subjects, samples, outputs, out_dir = epoch_vis_dir)

            visualizers['result'].visualize_all(subjects, samples, outputs, epoch_vis_dir, output_names = args.output_names, target_names = args.target_names) 
            if 'feature' in visualizers:
                visualizers['feature'].visualize_all_multi(subjects, samples, outputs, epoch_vis_dir)

    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger)) 

    if args.debug:
        exit()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}    




def train_one_epoch_downstream(epoch, args, feat_extractor, model, processors, 
                    criterion, data_loader, 
                    optimizer, lr_scheduler, wd_scheduler, feat_optimizer, feat_lr_scheduler, feat_wd_scheduler,
                    postprocessor = None, visualizers = None, output_dir = None,
                    device = 'cpu'):
    
    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(
        args.log_itr,
        delimiter="  ",
        debug=args.debug)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8f}'))

    header = 'Epoch: [{}/{}]'.format(epoch, args.n_epochs)

    for itr, (subjects, samples) in enumerate(metric_logger.log_every(data_loader, epoch, header=header, dataset_name=args.dataset, modality=args.modality, train_limit=args.train_itr_limit)): 
        
        # update weight decay and learning rate according to their schedule
        itr = len(data_loader) * epoch + itr  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_scheduler[itr]
            param_group["weight_decay"] = wd_scheduler[itr]
        if feat_optimizer is not None:
            for i, param_group in enumerate(feat_optimizer.param_groups):
                param_group["lr"] = feat_lr_scheduler[itr]
                param_group["weight_decay"] = feat_wd_scheduler[itr]

        samples = utils.nested_dict_to_device(samples, device)
        subjects = utils.nested_dict_to_device(subjects, device)

        # forward
        if args.freeze_feat:
            with torch.no_grad():
                feats, inputs = feat_extractor(samples)
        else:
            feats, inputs = feat_extractor(samples)

        outputs = model([feat['feat'] for feat in feats], inputs)
        for processor in processors:
            outputs = processor(outputs, samples)

        loss_dict = criterion(outputs, subjects, samples) 

        weight_dict = criterion.weight_dict 
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
 
        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            logger.info(f"Loss is {loss_value}, stopping training")
            logger.info(loss_dict_reduced)

        losses.backward()
        if args.clip_max_norm > 0:
            utils.clip_gradients(model, args.clip_max_norm)
        utils.cancel_gradients_last_layer(epoch, model, args.freeze_last_layer)

        optimizer.step()
        optimizer.zero_grad()

        if not args.freeze_feat:
            feat_optimizer.step()
            feat_optimizer.zero_grad()
        
        # logging
        if utils.get_world_size() > 1:
            torch.cuda.synchronize()
        metric_logger.update(loss = loss_value,
                            **loss_dict_reduced_scaled,
                            **loss_dict_reduced_unscaled
                            )
        metric_logger.update(lr = optimizer.param_groups[0]["lr"])
        metric_logger.update(wd = optimizer.param_groups[0]["weight_decay"]) 
        if not args.freeze_feat:
            metric_logger.update(lr_feat = feat_optimizer.param_groups[0]["lr"])
            metric_logger.update(wd_feat = feat_optimizer.param_groups[0]["weight_decay"]) 

        if args.debug or (itr % args.vis_itr == 0) and visualizers is not None and utils.is_main_process():
            epoch_vis_dir = utils.make_dir(os.path.join(output_dir, str(epoch), str(itr))) if epoch is not None else output_dir

            if postprocessor is not None:
                outputs = postprocessor(args, outputs, feats, task = args.task)
            if args.visualizer.make_results:  
                make_results(subjects, samples, outputs, out_dir = epoch_vis_dir)

            visualizers['result'].visualize_all(subjects, samples, outputs, epoch_vis_dir, output_names = args.output_names, target_names = args.target_names) 
            if 'feature' in visualizers:
                visualizers['feature'].visualize_all_multi(subjects, samples, outputs, epoch_vis_dir)

    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info("Averaged stats: {}".format(metric_logger)) 

    if args.debug:
        exit()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}    

