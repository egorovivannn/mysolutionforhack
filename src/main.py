from typing import Final 
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import requests
import torch
import albumentations as A
from model_utils import *
import cv2
import torch
import open_clip
import torch.nn.functional as F
from ultralytics import YOLO
import pandas as pd
import os
import json
import random
from PIL import Image

API_TOKEN: Final = '7481172726:AAFvlHhskIFhkpffMXtan72Gu2pWi1utHVw'
BOT_USERNAME: Final = '@welder_3000_bot'
global classifier, yolo_model, augs, idx2label

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_COMPLEX, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


async def predict(local_filename: str):
    pred_bboxes = yolo_model(local_filename, conf=0.05, iou=0.3, augment=True)
    img = pred_bboxes[0].orig_img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    if len(pred_bboxes[0].boxes.cls)==0:
        return {}
    else:
        imgs = []
        for bbox in pred_bboxes[0].boxes.xyxy:
            x1, y1, x2, y2 = bbox.cpu().tolist()
            x1_ = int(x1 // 1.2)
            y1_ = int(y1 // 1.2)
            x2_ = int(x2 * 1.2)
            y2_ = int(y2 * 1.2)       
            crop = img[y1_:y2_, x1_:x2_, :]
            crop = augs(image=crop)['image']
            crop = torch.from_numpy(crop).permute(2, 0, 1)
            imgs.append(crop)
        batch = torch.stack(imgs)
        
        with torch.no_grad():
            preds = classifier(batch.to(device))[0]
            preds = preds.cpu().detach().numpy().argmax(1).tolist()
        
        for bbox, pred in zip(pred_bboxes[0].boxes.xyxy, preds):
            plot_one_box(bbox.cpu().tolist(), img, color=(255, 0, 0), label=idx2label[pred])

        ans = {}
        for idx, bbox in enumerate(pred_bboxes[0].boxes.xywh):
            rel_x, rel_y, w_bbox, h_bbox = bbox.cpu().tolist()
            ans[idx] = {
                'filename': os.path.basename(local_filename),
                'rel_x': rel_x / w,
                'rel_y': rel_y / h,
                'w': w_bbox / w,
                'h': h_bbox / h,
                'class_id': preds[idx]
                
            }
        ans['img'] = img
        return ans


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Hello! I am a welder bot. Send me a photo of a welding seam and I will detect and classify your mistakes')

async def help_command (update: Update, context: ContextTypes. DEFAULT_TYPE):
    await update.message.reply_text('Send me a photo and I will respond')

async def custom_command(update: Update, context: ContextTypes. DEFAULT_TYPE):
    await update.message.reply_text( 'This is a custom command! ')




async def download_file(file_id: str, bot) -> str:
    file_info = await bot.get_file(file_id)
    print(file_info)
    url = file_info.file_path
    local_filename = './images/'+file_id + ".jpg"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        img_data = r.content
        with open(local_filename, 'wb') as handler:
                handler.write(img_data)
        # with open(local_filename, 'wb') as f:
        #     for chunk in r.iter_content(chunk_size=8192):
        #         f.write(chunk)
    return local_filename



async def handle_message_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    local_filename = './images/temp.png'
    file2 = await context.bot.get_file(update.message.document)
    await file2.download_to_drive(local_filename)


    await update.message.reply_text('Image received and is being processed')
    predicted_labels = await predict(local_filename)
    if not predicted_labels:
        await update.message.reply_text('Your image has no defects')
    else:
        img = predicted_labels.pop('img')
        Image.fromarray(img).save('./images/temp.png')
        img = open('./images/temp.png', 'rb')
        with open('ans.json', 'w') as f:
            json.dump(predicted_labels, f)

        predicted_labels = open('ans.json', 'rb')
        await update.message.reply_text('Your prediction is:')
        await update.message.reply_photo(img)
        await update.message.reply_document(predicted_labels)





async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) :
    print(f'Update {update} caused error {context.error}')






if __name__=='__main__':
    idx2label = {
        0: 'прилегающие дефекты', 
        1: 'дефекты целостности',
        2: 'дефекты геометрии',
        3: 'дефекты постобработки',
        4: 'дефекты невыполнения'
    }


    yolo_model = YOLO('./checkpoints/best.pt')

    model_name = 'ViT-H-14-378-quickgelu'
    pretrained = 'dfn5b'
    device = 'cuda'

    vit_backbone, model_transforms, _ = open_clip.create_model_and_transforms(model_name, pretrained=False)

    mean, std = model_transforms.transforms[-1].mean, model_transforms.transforms[-1].std
    image_size = model_transforms.transforms[0].size[0]

    classifier = Model(vit_backbone.cpu(), image_size).to(device)
    classifier.load_state_dict(torch.load('./checkpoints/ViT-H-14-378-quickgelu_dfn5b_0.7920078574145561.pth'))
    classifier.eval()


    augs = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std, p=1)
        ])



    print( 'Starting bot...')
    app = Application.builder().token(API_TOKEN).build()

    app.add_handler (CommandHandler ('start', start_command))
    app.add_handler (CommandHandler ('help', help_command)) 
    app.add_handler (CommandHandler ('custom', custom_command))
    
    app.add_handler(MessageHandler(filters.Document.IMAGE, handle_message_photo))
    app.add_error_handler(error)
    app.run_polling(poll_interval=3)

