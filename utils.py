from models.resnet import resnet20, resnet56
def GetModel(model, pretrained):
    if model == "resnet20":
        net = resnet20(num_classes=100,pretrained=pretrained)
    elif model == "resnet56":
        net = resnet56(num_classes=100,pretrained=pretrained)
    return net