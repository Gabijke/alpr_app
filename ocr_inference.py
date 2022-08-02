import numpy as np
import torch
from torchvision import transforms
from ocr_model import OcrNet
from deskew_license_plate import deskew_img


def decoder_predict(text):
    text = list(text)
    blank_char = '-'

    for i in range(len(text)):
        for j in range(i + 1, len(text)):
            if text[j] == blank_char:
                break
            else:
                if text[j] == text[i]:
                    text[j] = blank_char
                else:
                    continue
    final_text = ''.join(text).replace(blank_char, '')

    return final_text


def predict(img):
    alphabet = "-1234567890ABEKMHOPCTYX"
    PATH = 'models/OcrNet.pth'
    model = OcrNet()
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Grayscale(),
                                transforms.Resize((50, 200)),
                                transforms.ToTensor()])

    if isinstance(img, np.ndarray):
        img = deskew_img(img)
        X = trans(img).unsqueeze(0)
        model.cpu()
        X = X.to(device)
        target = model(X)
        _, target = target.max(2)
        for prediction in target.permute(1, 0):
            prediction_text = ''.join([alphabet[i.item()] for i in prediction])
            final_predict = decoder_predict(prediction_text)
            return final_predict
