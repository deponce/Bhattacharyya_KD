import torch 
import torch.nn as nn
import torchvision

def CurrectlyClassified(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.inference_mode():
        maxk = max(topk)
        # batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k)
        return res

class KnowledgeDistilldation(object):
    def __init__(self, Teacher, Student, Optimizer, TrainLoader, ValLoader, device, LossFn, LossIndicator, Lambda) -> None:
        self.Teacher = Teacher
        self.Student = Student
        self.LossFn = LossFn
        self.Lambda = Lambda
        self.__LossIndicator = LossIndicator
        self.__TeacherInList = False
        self.__StudentInList = False
        self.Optimizer = Optimizer
        self.TrainLoader = TrainLoader
        self.ValLoader = ValLoader
        self.device = device
        if not self.device:
            self.device = torch.device('cpu')
        assert isinstance(self.LossFn, list), f"Please provide a list of Lambdas."
        assert isinstance(self.Lambda, list), f"Please provide a list of loss functions."
        assert isinstance(self.__LossIndicator, list), f"Please provide a list of loss indicator."
        uniLossIndicator = list(set(self.__LossIndicator))
        assert len(uniLossIndicator) == 2 and 'H' in uniLossIndicator and 'S' in uniLossIndicator, f"LossIndicator only contain 'H' and 'S'."
        assert len(LossFn) == len(self.Lambda), f"Number of LossFn should match number of Lambda. However we get : {len(self.Lambda)} Lamda"
        assert len(LossFn) == len(self.__LossIndicator), f"Number of LossFn should match number of loss indicator. However we get : {len(self.Lambda)} Lamda"
        if isinstance(self.Teacher, list): self.__TeacherInList = True
        if isinstance(self.Student, list): self.__StudentInList = True

    def UpdateLambda(self, Lambda):
        assert isinstance(Lambda, list), f"The Lambda should be a list."
        assert len(Lambda) == len(self.Lambda), f"Length of Lambda mismatch the original one, get length {len(Lambda)}, but the original one with {len(self.Lambda)}."
   
    def TrainOneEpoch(self):
        Nsamples = 0
        # TODO: didn't consider multi teacher case
        self.Teacher.eval()
        self.Student.Train()
        Tncc1 = 0
        Tncc5 = 0
        for data, target in self.TrainLoader:
            Nsamples += target.size(0)
            data,target = data.to(self.device), target.to(self.device)
            self.Optimizer.zero_grad()
            if self.__TeacherInList: TeacherOutput = [t(data) for t in self.Teacher]
            else: TeacherOutput = self.Teacher(data)
            if self.__StudentInList: StudentOutput = [s(data) for s in self.Student]
            else: StudentOutput = self.Student(data)
            Losses = [None for _ in len(self.LossFn)]
            RtLosses = [0 for _ in len(self.LossFn)]
            if not(self.__TeacherInList and self.__TeacherInList):
                ncc1, ncc5 = CurrectlyClassified(StudentOutput, target, (1,5))
                Tncc1+=ncc1
                Tncc5+=ncc5
                LossesForEachFn = []
                for idx, lfn in enumerate(LossesForEachFn):
                    if self.__LossIndicator[idx] == "H":
                        LossVal = lfn(target, StudentOutput)
                        
                    elif self.__LossIndicator[idx] == "S":
                        LossVal = lfn(TeacherOutput, StudentOutput)
                    else:
                        print('unexpected in self.__LossIndicator', flush=True)
                        exit(-1)
                    Losses[idx] = LossVal
                    RtLosses[idx] += float(LossVal)*target.size(0)
            else:
                pass # waiting for implement
            # if self.__StudentInList:  waiting for implement
            Loss = None
            for idx, l in enumerate(Losses):
                if not Loss: Loss = l*self.Lambda[idx]
                else: Loss += l*self.Lambda[idx]
            Loss.backward()
            self.Optimizer.step()
        RtLosses = [l/Nsamples for l in RtLosses]
        return {'Losses':RtLosses, 'Top1':Tncc1/Nsamples, "Top5": Tncc5/Nsamples}
    
    def Validation(self):
        Nsamples = 0
        self.Student.eval()
        Tncc1 = 0
        Tncc5 = 0
        RtLosses = [0 for _ in len(self.LossFn)]
        for data, target in self.ValLoader:
            Nsamples += target.size(0)
            data,target = data.to(self.device), target.to(self.device)
            StudentOutput = self.Student(data)
            ncc1, ncc5 = CurrectlyClassified(StudentOutput, target, (1,5))
            Tncc1+=ncc1
            Tncc5+=ncc5
            for idx, lfn in enumerate(self.LossFn):
                if self.__LossIndicator[idx] == "H":
                    LossVal = lfn(target, StudentOutput)
                    RtLosses[idx]+=float(LossVal)*target.size(0)
        RtLosses = [l/Nsamples for l in RtLosses]
        return {'Losses':RtLosses, 'Top1':Tncc1/Nsamples, "Top5": Tncc5/Nsamples}
    