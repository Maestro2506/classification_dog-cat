import os
import base64
import json
import requests
import pandas as pd
import argparse

url = 'http://127.0.0.1:5000/predict'

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='images', help='specify the name of the folder where your images are saved')
parser.add_argument('--excel_name', type = str, required=False, help='specify the name of excel file if you want to crrate new excel file with ypur data')
args = parser.parse_args()

main_dict = {"photos":[]}
directory = args.directory
true_labels = []

print("Making predictions... It may take several minutes!")

for label in os.listdir(directory):
  for image in os.listdir(os.path.join(directory,label)):
    with open(os.path.join(directory,label,image), "rb") as image_file:
      encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
      true_labels.append(0 if label=='cat' else 1)
      dict = {"ID":label+image.split('.')[1],"img_code":encoded_image}
      main_dict["photos"].append(dict)

headers = {'content-type': 'application/json'}
response = requests.post(url, data=json.dumps(main_dict), headers=headers)
df = pd.DataFrame(response.json()['results'])
df['true_label'] = true_labels
df[['cat_prob', 'dog_prob']] = df[['cat_prob', 'dog_prob']].astype(float)
if args.excel_name is None:
  df.to_excel("probabilities.xlsx", index = False)

print("Finished")