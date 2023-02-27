from models.resnet import resnet20, resnet56
def GetModel(model):
    if model == "resnet20":
        net = resnet20(num_classes=100,pretrained=True)
    elif model == "resnet56":
        net = resnet56(num_classes=100,pretrained=True)
    return net