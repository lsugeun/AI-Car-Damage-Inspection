import torch
import cv2
import matplotlib.pyplot as plt
import base64
import numpy as np

from flask import Flask, render_template, request

from src.Models import Unet

app = Flask(__name__)

# 여러 형태의 파손 영역 감지
labels = ['Breakage_3', 'Crushed_2', 'Scratch_0', 'Seperated_1']
models = []

n_classes = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

for label in labels:
    model_path = f'[DAMAGE][{label}]Unet.pt'

    model = Unet(encoder='resnet34', pre_weight='imagenet', num_classes=n_classes).to(device)
    model.model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()

    models.append(model)

print('Loaded pretrained models!')

# 이미지 업로드 및 결과 표시
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

        # 이미지 크기 조정
        img = cv2.resize(img, (256, 256))

        img_input = img / 255.
        img_input = img_input.transpose([2, 0, 1])
        img_input = torch.tensor(img_input).float().to(device)
        img_input = img_input.unsqueeze(0)

        fig, ax = plt.subplots(1, len(labels) + 1, figsize=(24, 10))

        ax[0].imshow(img)
        ax[0].axis('off')

        outputs = []

        for i, model in enumerate(models):
            output = model(img_input)

            img_output = torch.argmax(output, dim=1).detach().cpu().numpy()
            img_output = img_output.transpose([1, 2, 0])
            
            outputs.append(img_output)

            ax[i+1].set_title(labels[i])
            ax[i+1].imshow(img_output, cmap='jet')
            ax[i+1].axis('off')

        fig.set_tight_layout(True)

        # 이미지를 Base64로 인코딩하여 전달
        img_encoded = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode('utf-8')

        # 파손 영역의 크기와 가격 계산
        repair_details = []

        price_table = [
            100,  # Breakage_3
            200,  # Crushed_2
            50,   # Scratch_0
            120   # Seperated_1
        ]

        for i, label in enumerate(labels):
            area = outputs[i].sum()
            price = price_table[i]
            price_total = area * price

            repair_details.append({'label': label, 'area': area, 'price': price_total})

        total = sum(detail['price'] for detail in repair_details)

        return render_template('index.html', image=img_encoded, price=total, repair_details=repair_details)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
