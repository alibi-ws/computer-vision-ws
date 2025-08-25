from ultralytics import YOLO
import xml.etree.ElementTree as ET
import os
import shutil
from sklearn.model_selection import train_test_split
import kagglehub


# Download latest version
path = kagglehub.dataset_download("andrewmvd/face-mask-detection")

print("Path to dataset files:", path)

def convert_XML2YOLO(xml_dir, output_dir, xml_file, classes):
  tree = ET.parse(os.path.join(xml_dir, xml_file))
  root = tree.getroot()

  # Extract image size for normalization for yolo format
  image_width = int(root.find("size/width").text)
  image_height = int(root.find("size/height").text)

  file_name = root.find('filename').text
  file_name = file_name.split('.')[0]
  file_name = file_name + '.txt'
  label_path = os.path.join(output_dir, file_name)

  with open(label_path, 'w') as f:
    for obj in root.findall('object'):
      cls = obj.find('name').text
      cls_id = classes.index(cls)
      xml_box = obj.find('bndbox')
      xmin = int(xml_box.find("xmin").text)
      ymin = int(xml_box.find("ymin").text)
      xmax = int(xml_box.find("xmax").text)
      ymax = int(xml_box.find("ymax").text)

      # Convert to YOLOv8 format
      x_center = ((xmin + xmax) / 2) / image_width
      y_center = ((ymin + ymax) / 2) / image_height
      w = (xmax - xmin) / image_width
      h = (ymax - ymin) / image_height

      f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

# Convert XML files to yolov8 version
classes = ["without_mask", "with_mask", "mask_weared_incorrect"]
xml_dir = path + '/annotations'
output_dir = '/kaggle/working/labels'
os.makedirs(output_dir, exist_ok=True)

for xml_file in os.listdir(xml_dir):
  if not xml_file.endswith('.xml'):
    continue

  convert_XML2YOLO(xml_dir, output_dir, xml_file, classes)

# Copy folder from kaggle input to working
images_new_dir = "/kaggle/working/images"
shutil.copytree(path+'/images', images_new_dir)

all_images = os.listdir('/kaggle/working/images')

os.makedirs('/kaggle/working/train/images', exist_ok=True)
os.makedirs('/kaggle/working/train/labels', exist_ok=True)
os.makedirs('/kaggle/working/val/images', exist_ok=True)
os.makedirs('/kaggle/working/val/labels', exist_ok=True)

train, val = train_test_split(all_images, test_size=0.2, random_state=42,
                              shuffle=True)

for img in train:
  shutil.copy(images_new_dir+'/'+img, '/kaggle/working/train/images')
  shutil.copy(output_dir+'/'+img.split('.')[0]+'.txt', '/kaggle/working/train/labels')

for img in val:
  shutil.copy(images_new_dir+'/'+img, '/kaggle/working/val/images')
  shutil.copy(output_dir+'/'+img.split('.')[0]+'.txt', '/kaggle/working/val/labels')

yaml_content = '''
train: /kaggle/working/train/images
val: /kaggle/working/val/images
nc: 3
names: ["with_mask", "without_mask", "mask_weared_incorrect"]
'''
with open('/kaggle/working/data.yaml', 'w') as f:
  f.write(yaml_content)

model = YOLO('yolov8n.pt')

print("Starting initial training (full model)...")
model.train(
    data='/kaggle/working/data.yaml',
    epochs=5,
    batch=64,
    imgsz=640,
    lr0=1e-3,      # initial learning rate
    cos_lr=True,  # LR decay
    workers=4,
    project="mask_yolov8",
    name="full_model",
    exist_ok=True,
    augment=True,  # use augmentation
)

print("Starting fine tuning \n")
model.train(
    data='/kaggle/working/data.yaml',
    epochs=5,
    batch=64,
    imgsz=640,
    lr0=1e-3,      # initial learning rate
    cos_lr=True,  # LR decay
    workers=4,
    freeze=10,
    project="mask_yolov8",
    name="full_model",
    exist_ok=True,
    augment=True,  # use augmentation
)

model.val()
results = model.predict(source='/kaggle/input/face-mask-detection/images/maksssksksss0.png', save=True)

