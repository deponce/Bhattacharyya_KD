import torch.nn.functional as F

class SoftLoss():
    def __init__(self, T=4) -> None:
        self.T = T
    def __call__(self,preds, teacher_preds):
         
        loss = (self.T**2)*F.kl_div(F.log_softmax(preds / self.T, dim=1), 
                        F.softmax(teacher_preds / self.T, dim=1), reduction='batchmean')
        return loss