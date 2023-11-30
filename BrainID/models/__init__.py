

"""
Submodule interface.
"""
import torch


from .backbone import build_backbone
from .criterion import *
from .evaluator import Evaluator
from .head import get_head
from .joiner import get_processors, get_joiner
import utils.misc as utils


#########################################

# some constants
label_list_segmentation = [0, 14, 15, 16, 24, 77, 85, 2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 41,
                                    42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
n_neutral_labels = 7
n_labels = len(label_list_segmentation)
nlat = int((n_labels - n_neutral_labels) / 2.0)
vflip = np.concatenate([np.array(range(n_neutral_labels)),
                        np.array(range(n_neutral_labels + nlat, n_labels)),
                        np.array(range(n_neutral_labels, n_neutral_labels + nlat))]) 

                        
############################################
############# helper functions #############
############################################

def process_args(args, task = 'feat'):
    """
    task options: feat-anat, feat-seg, seg, reg, sr, bf
    """
    args.base_generator.size = args.base_generator.sample_size # update real sample size (if sample_size is downsampled)

    args.n_labels_with_csf = len(args.base_generator.label_list_segmentation_with_csf)
    args.n_labels_without_csf = len(args.base_generator.label_list_segmentation_without_csf)

    args.out_channels = {}
    args.output_names = []
    args.aux_output_names = []
    args.target_names = []
    if not 'contrastive' in task:
        if 'anat' in task:
            args.out_channels['image'] = 2 if args.losses.uncertainty is not None else 1
            args.output_names += ['image']
            args.target_names += ['image']
            if args.losses.uncertainty is not None:
                args.aux_output_names += ['image_sigma']
        if 'sr' in task:
            args.out_channels['image'] = 2 if args.losses.uncertainty is not None else 1
            args.output_names += ['image']
            args.target_names += ['orig']
            if args.losses.uncertainty is not None:
                args.aux_output_names += ['image_sigma']
        if 'bf' in task:
            args.out_channels['bias_field_log'] = 2 if args.losses.uncertainty is not None else 1
            args.output_names += ['bias_field_log']
            args.target_names += ['bias_field_log']
        if 'seg' in task:
            args.out_channels['seg'] = args.n_labels_with_csf 
            args.output_names += ['label']
            args.target_names += ['label']
            
        assert len(args.output_names) > 0

    return args

############################################
################ CRITERIONS ################
############################################

def get_evaluator(args, task, device):
    """
    task options: sr, seg, anat, reg
    """
    metric_names = []
    if 'feat' in task:
        metric_names += ['feat_ssim', 'feat_ms_ssim', 'feat_l1']
    else:
        if 'anat' in task:
            metric_names += ['recon_l1', 'recon_psnr', 'recon_ssim', 'recon_ms_ssim']
        if 'sr' in task:
            metric_names += ['sr_l1', 'sr_psnr', 'sr_ssim', 'sr_ms_ssim']
        if 'bf' in task: 
            metric_names += ['bf_normalized_l2', 'bf_corrected_l1']
        if 'seg' in task:
            metric_names += ['seg_dice']
        
    assert len(metric_names) > 0

    evaluator = Evaluator(
        args = args,
        metric_names = metric_names, 
        device = device,
        )
        
    return evaluator



def get_criterion(args, task, device):
    """
    task options: sr, seg, anat, reg
    """
    loss_names = []
    weight_dict = {}

    if 'contrastive' in task:
        loss_names += ['contrastive']
        weight_dict['loss_contrastive'] = args.weights.contrastive
        return SetCriterion(
            args = args,
            weight_dict = weight_dict,
            loss_names = loss_names, 
            device = device,
            )
    
    if 'anat' in task or 'sr' in task: 
        name = 'sr' if 'sr' in task else 'image'

        loss_names += [name]
        weight_dict.update({'loss_%s' % name: args.weights.image})
        if args.losses.image_grad:
            loss_names += ['%s_grad' % name]
            weight_dict['loss_%s_grad' % name] = args.weights.image_grad 
    if 'seg' in task:
        loss_names += ['seg_ce', 'seg_dice']
        weight_dict.update( {
            'loss_seg_ce': args.weights.seg_ce,
            'loss_seg_dice': args.weights.seg_dice,
        } )
    if 'bf' in task:
        loss_names += ['bias_field_log']
        weight_dict.update( {
            'loss_bias_field_log': args.weights.bias_field_log, 
        } )
    if 'reg' in task:
        loss_names += ['reg', 'reg_grad']
        weight_dict['loss_reg'] = args.weights.reg
        weight_dict['loss_reg_grad'] = args.weights.reg_grad
        
    assert len(loss_names) > 0

    criterion = SetMultiCriterion(
        args = args,
        weight_dict = weight_dict,
        loss_names = loss_names, 
        device = device,
        )
        
    return criterion




def get_postprocessor(args, outputs, feats = None, task = 'seg'):
    """
    output: list of output dict 
    feat: list of output dict from pre-trained feat extractor
    """
    for i, output in enumerate(outputs): 
        if feats is not None:
            output.update({'feat': feats[i]['feat']}) 
        if 'seg' in task:
            #output['label'] = torch.tensor(args.base_generator.label_list_segmentation_with_csf, 
            #                             device = output['seg'].device)[torch.argmax(output['seg'][:, vflip], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 
            output['label'] = torch.tensor(args.base_generator.label_list_segmentation_with_csf, 
                                         device = output['seg'].device)[torch.argmax(output['seg'], 1, keepdim = True)] # (b, n_labels, s, r, c) -> (b, s, r, c) 
    return outputs


#############################################
################ OPTIMIZERS #################
#############################################


def build_optimizer(args, params_groups):
    if args.optimizer == "adam":
        return torch.optim.Adam(params_groups)  
    elif args.optimizer == "adamw":
        return torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        return torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        return utils.LARS(params_groups)  # to use with convnet and large batches
    else:
        ValueError('optim type {args.optimizer.type} supported!')


def build_schedulers(args, itr_per_epoch, lr, min_lr):
    if args.lr_scheduler == "cosine":
        lr_scheduler = utils.cosine_scheduler(
            lr, # * (args.batch_size * utils.get_world_size()) / 256.,  # linear scaling rule
            min_lr,
            args.n_epochs, itr_per_epoch,
            warmup_epochs=args.warmup_epochs
        )
    elif args.lr_scheduler == "multistep":
        lr_scheduler = utils.multistep_scheduler(
            lr, 
            args.lr_drops, 
            args.n_epochs, itr_per_epoch, 
            warmup_epochs=args.warmup_epochs, 
            gamma=args.lr_drop_multi
            )  
    wd_scheduler = utils.cosine_scheduler(
        args.weight_decay, # set as 0 to disable it
        args.weight_decay_end,
        args.n_epochs, itr_per_epoch
        )
    return lr_scheduler, wd_scheduler


############################################
################## MODELS ##################
############################################


def build_feat_model(args, device = 'cpu'):
    args = process_args(args, task = args.task)

    backbone = build_backbone(args)
    head = get_head(args, args.task_f_maps, args.out_channels, True, -1)
    model = get_joiner(args.task, backbone, head) 

    processors = get_processors(args, args.task, device)

    criterion = get_criterion(args, args.task, device)
    criterion.to(device)

    model.to(device) 
    postprocessor = get_postprocessor

    return args, model, processors, criterion, postprocessor



def build_downstream_model(args, device = 'cpu'):
    args = process_args(args, task = args.task)

    backbone = build_backbone(args)

    feat_model = get_joiner(args.task, backbone, None) 
    task_model = get_head(args, args.task_f_maps, args.out_channels, True, -1)

    processors = get_processors(args, args.task, device)

    criterion = get_criterion(args, args.task, device)
    criterion.to(device)

    feat_model.to(device) 
    task_model.to(device)
    postprocessor = get_postprocessor

    return args, feat_model, task_model, processors, criterion, postprocessor
