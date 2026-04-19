import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import functools
import string


class BidirectionalLSTM(nn.Module):
    def __init__(self,in_size,hidden,out_size):
        super().__init__()
        self.lstm=nn.LSTM(in_size,hidden,bidirectional=True,dropout=0.2,batch_first=False)
        self.fc=nn.Linear(hidden*2,out_size)

    def forward(self,x):
        out,_=self.lstm(x)
        T,B,C=out.shape
        return self.fc(out.view(T*B,C)).view(T,B,-1)

class CRNN(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(1,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128),nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout2d(0.2),
            nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout2d(0.2),
            nn.Conv2d(512,512,2), nn.BatchNorm2d(512), nn.ReLU()
        )
        self.rnn=nn.Sequential(
            BidirectionalLSTM(512,256,256),
            BidirectionalLSTM(256,256,num_classes)
        )

    def forward(self, x):
      feat = self.cnn(x)
      feat = feat.squeeze(2)
      feat = feat.permute(2, 0, 1)
      return self.rnn(feat)
    
CONFIG={
    "model_path": os.path.join(os.getcwd(), "MODEL", "CRNN_best_model.pth"),
    "img_h":32,
    "img_w":128,
    "char_set":string.ascii_letters + string.digits
}

@functools.lru_cache(maxsize=1)
def _load_model():
    """
    Loads model once and caches it for subsequent calls
    """

    num_classes=len(CONFIG["char_set"])+1

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=CRNN(num_classes).to(device)
    model.load_state_dict(torch.load(CONFIG["model_path"],map_location=device))

    model.eval()

    print(f"CRNN Model loaded on {device}.")
    return model, device

def _preprocess(image_path:str) -> torch.Tensor:
    """
    preprocess image for model input
    """
    img=Image.open(image_path).convert("L")
    transform=transforms.Compose([
        transforms.Resize((CONFIG["img_h"], CONFIG["img_w"])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return transform(img).unsqueeze(0)

def _ctc_decode(preds:torch.Tensor)->str:
    """
    Decodes model output using CTC decoding rules
    """
    chars=CONFIG["char_set"]
    indicies=preds.squeeze(1).argmax(dim=1).tolist()

    decoded, prev=[],None
    for idx in indicies:
        if idx !=prev and idx !=0:
            decoded.append(chars[idx-1])
        prev=idx
    
    return "".join(decoded)

def predict_captcha(image_path:str)->str:
    """
    Takes image path, returns predicted CAPTCHA string
    Model is loaded once and cached for all subsequent calls."""

    model,device=_load_model()
    tensor=_preprocess(image_path).to(device)

    with torch.no_grad():
        logits=model(tensor)
        log_probs=logits.log_softmax(2)
    
    return _ctc_decode(log_probs)

result=predict_captcha("/home/gautam/rcnn captcha/captcha_dataset/state_portal_captchas/Q6OSAl.png")
print(f"Predicted CAPTCHA: {result}")