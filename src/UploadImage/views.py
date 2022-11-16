import io
import os
import json
import torch

from torchvision import models
from torchvision import transforms
from PIL import Image
from django.conf import settings

from django.shortcuts import render
from .forms import ImageForm
import base64

model = models.densenet121(pretrained=True)
#model.load_state_dict(torch.load(os.path.join(settings.STATIC_ROOT, "state_dict_model.pt")))
model.eval()

json_path=os.path.join(settings.STATIC_ROOT, "imagenet_class_index.json")
imagenet_mapping = json.load(open(json_path))

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                           [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225] 
                                        )])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
    img_tensor = transform_image(image_bytes)
    outputs = model.forward(img_tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    class_name, human_label = imagenet_mapping[predicted_idx]
    return human_label

def image_upload_view(request):
    image_uri = None
    predicted_label = None
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            image_bytes = image.file.read()
            encoded_img = base64.b64encode(image_bytes).decode('ascii')
            image_uri = 'data:%s;base64,%s' % ('image/jpeg', encoded_img)

            try:
                predicted_label = get_prediction(image_bytes)
            except RuntimeError as re:
                print(re)

    else:
        form = ImageForm()

    context = {
        'form': form,
        'image_uri': image_uri,
        'predicted_label': predicted_label,
    }
    return render(request, 'index.html', context)
