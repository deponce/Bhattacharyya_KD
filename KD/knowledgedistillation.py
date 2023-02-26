import torch 
import torch.nn as nn
import torchvision
from torchvision.models import resnet50

class KnowledgeDistilldation(object):
    def __init__(self, Teacher, Student, Optimizer, TrainLoader, ValLoader, device, LossFn, Lambda) -> None:
        self.Teacher = Teacher
        self.Student = Student
        self.LossFn = LossFn
        self.Lambda = Lambda
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
        assert len(LossFn) == len(Lambda), f"Number of LossFn should match number of Lambda. However we get : {len(self.Lambda)} Lamda"
        if isinstance(self.Teacher, list): self.__TeacherInList = True
        if isinstance(self.Student, list): self.__StudentInList = True
    def UpdateLambda(self, Lambda):
        assert isinstance(Lambda, list), f"The Lambda should be a list."
        assert len(Lambda) == len(self.Lambda), f"Length of Lambda mismatch the original one, get length {len(Lambda)}, but the original one with {len(self.Lambda)}."
    def TrainOneEpoch(self,):
        for data, target in self.TrainLoader:
            data,target = data.to(self.device), target.to(self.device)
            if self.__TeacherInList: TeacherOutput = [t(data) for t in self.Teacher]
            else: TeacherOutput = self.Teacher(data)
            if self.__StudentInList: StudentOutput = [s(data) for s in self.Student]
            else: StudentOutput = self.Student(data)
            
model = resnet50()
breakpoint()