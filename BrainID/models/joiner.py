

"""
Wrapper interface.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


class SegProcessor(nn.Module):
    def __init__(self):
        super(SegProcessor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['seg'] = self.softmax(output['seg'])
        return outputs
    

class ContrastiveProcessor(nn.Module):
    def __init__(self):
        '''
        Ref: https://openreview.net/forum?id=2oCb0q5TA4Y
        '''
        super(ContrastiveProcessor, self).__init__()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, outputs, *kwargs):
        for output in outputs:
            output['feat'][-1] = F.normalize(output['feat'][-1], dim = 1)
        return outputs


class BFProcessor(nn.Module):
    def __init__(self):
        super(BFProcessor, self).__init__()

    def forward(self, outputs, *kwargs): 
        for output in outputs:
            output['bias_field'] = torch.exp(output['bias_field_log'])
        return outputs



##############################################################################


class MultiInputIndepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    """
    def __init__(self, backbone, head):
        super(MultiInputIndepJoiner, self).__init__()

        self.backbone = backbone 
        self.head = head

    def forward(self, input_list):
        outs = []
        for x in input_list:  
            feat = self.backbone.get_feature(x['input'])
            out = {'feat': feat}
            if self.head is not None: 
                out.update( self.head(feat) )
            outs.append(out)
        return outs, [input['input'] for input in input_list]


class MultiInputDepJoiner(nn.Module):
    """
    Perform forward pass separately on each augmented input.
    """
    def __init__(self, backbone, head):
        super(MultiInputDepJoiner, self).__init__()

        self.backbone = backbone 
        self.head = head

    def forward(self, input_list):
        outs = []
        for x in input_list:  
            feat = self.backbone.get_feature(x['input'])
            out = {'feat': feat} 
            if self.head is not None: 
                out.update( self.head( feat, x['input']) )
            outs.append(out)
        return outs, [input['input'] for input in input_list]



################################


def get_processors(args, task, device):
    processors = []
    if 'contrastive' in task:
        processors.append(ContrastiveProcessor().to(device))
    if 'seg' in task:
        processors.append(SegProcessor().to(device))
    if 'bf' in task:
        processors.append(BFProcessor().to(device))
    return processors


def get_joiner(task, backbone, head):
    if 'sr' in task or 'bf' in task:
        return MultiInputDepJoiner(backbone, head) 
    else:
        return MultiInputIndepJoiner(backbone, head)