import json
import os
import time
import cv2
import torch
from PIL import Image
from transformers import BlipProcessor,BlipForConditionalGeneration
import  shutil


def resize_image(input_folder,output_folder):
    '''resize image '''
    new_size = (256, 256)
    image_list = sorted(os.listdir(input_folder))
    for i in range (len(image_list)):
        input_path = os.path.join(input_folder,image_list[i])
        output_path = os.path.join(output_folder,image_list[i])
        image = cv2.imread(input_path)
        resized_image = cv2.resize(image,new_size,interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path,resized_image)
    print("完成")

def blip(image_path):
     '''image2text model'''
     # 1. load pretrain model
     processor = BlipProcessor.from_pretrained("/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/pretrain_model/blip-image-captioning-base")
     model = BlipForConditionalGeneration.from_pretrained("/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/pretrain_model/blip-image-captioning-base",
                                                          torch_dtype=torch.float16).to("cuda:0")
    # 2. convert image to RGB
     raw_image = Image.open(image_path).convert('RGB')

     # conditional image captioning
     text = "an infrared image of"   # condition text
     inputs = processor(raw_image, text, return_tensors="pt").to("cuda:0", torch.float16)  # input image
     out = model.generate(**inputs)  # get out image
     text_str = processor.decode(out[0], skip_special_tokens=True)
     print('\033[91m'+ processor.decode(out[0], skip_special_tokens=True)+'\033[0m')
     '''
     #>>> a photography of a woman and her dog

     # # # # unconditional image captioning
     # inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
     # out = model.generate(**inputs)
     # # print(processor.decode(out[0], skip_special_tokens=True))
     # text_str = processor.decode(out[0], skip_special_tokens=True)
     # print('\033[94m'+'image describe:' + processor.decode(out[0], skip_special_tokens=True) + '\033[0m')
     '''
     return text_str

def imagetotext(image_path,out_put_path):
        '''image2text fuction of run model and save prompt'''
        text = blip(image_path)
        text_path = os.path.join(out_put_path, image_path.split('/')[-1].split('.') [0]+ '.txt')
        print(text_path)
        with open(text_path,'w') as file:
            file.write(text)


def generator_json_file(source_file,target_file,prompt_text,output_json_path):
     '''
     source file : VIS image path
     target file : thermal/infrared image path
     prompt_text : Description of target image
     output_json_path : json output path
     '''
     data_list = []

     for i in range(1):
         assert  source_file.split('/')[-1].split('.')[0] == target_file.split('/')[-1].split('.')[0] == prompt_text.split('/')[-1].split('.')[0]
         text_path = prompt_text
         # 读取第一行并转换为字符串
         with open(text_path, 'r') as file:
             first_line = file.readline().strip()
         data = {
              "source": source_file,
              "target": target_file,
              "prompt": first_line
         }
         data_list.append(data)
         output_json_path = os.path.join(output_json_path,target_file.split('/')[-1].split('.')[0]+'.json')
     with open(output_json_path,'w') as json_file:
         # json_file.write('[')
         for i,data in enumerate(data_list):
             json_line = json.dumps(data)
             # if i < len(data_list)-1:
             #     json_file.write(json_line + ',' + '\n')
             # else:
             json_file.write(json_line + '\n')

         # json_file.write(']')
     print(f"数据列表已成功写入 JSON 文件：{output_json_path}")
     return output_json_path

###########################################################
'''
image process
'''
def train_image_look(input_image_file,output_file):
    input_image_list = sorted(os.listdir(input_image_file))
    assert len(input_image_list) % 4 == 0
    image_number  = int(len(input_image_list)/4)
    for i in range(image_number):
        out_file  = os.path.join(output_file,str(i))
        os.makedirs(out_file,exist_ok=True)
        for j in range(len(input_image_list)):
            if "_".join(input_image_list[i].split('_')[1:])in input_image_list[j]:
                source_file = os.path.join(input_image_file,input_image_list[j])
                target_file = os.path.join(out_file,input_image_list[j])
                shutil.copy(source_file,target_file)
            else:
                pass
        print('file {} is complete!'.format(i+1))
    print('End split')


def replace_str(json_file_path):
    count = 0
    with open(json_file_path,'r') as file:
        lines  = file.readlines()
    # 遍历每一行
    for i in range(len(lines)):
        # 尝试将每一行的 JSON 字符串转换为 Python 字典
        try:
            json_object = json.loads(lines[i])

            # 判断 "prompt" 键是否存在，并且包含 "at night" 字符串
            if 'prompt' in json_object and 'at night' in json_object['prompt']:
                # 如果存在，将 "at night" 替换为空字符串
                json_object['prompt'] = json_object['prompt'].replace('at night', '')
                count += 1
                # 将修改后的 JSON 字符串写回到列表
                lines[i] = json.dumps(json_object,indent=None)+'\n'
        except json.JSONDecodeError:
            # 如果无法解析为 JSON，跳过该行
            continue
    print("total replace number is:{}".format(count))

    # 将修改后的内容写回文件
    with open(json_file_path, 'w') as file:
        file.writelines(lines)

def replace_prefix(input_json_file_path,out_json_file,prefix_str):
    count = 0
    with open(input_json_file_path,'r') as file:
        lines  = file.readlines()
    # 遍历每一行
    for i in range(len(lines)):
        # 尝试将每一行的 JSON 字符串转换为 Python 字典
        try:
            json_object = json.loads(lines[i])

            # 判断 "prompt" 键是否存在，并且包含 "at night" 字符串
            if 'prompt' in json_object and prefix_str in json_object['prompt']:
                # 如果存在，将 "at night" 替换为空字符串
                json_object['prompt'] = json_object['prompt'].replace(prefix_str, 'an infrared of')
                count += 1
                # 将修改后的 JSON 字符串写回到列表
                lines[i] = json.dumps(json_object,indent=None)+'\n'
        except json.JSONDecodeError:
            # 如果无法解析为 JSON，跳过该行
            continue
    print("total replace number is:{}".format(count))

    # 将修改后的内容写回文件
    with open(out_json_file, 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    generator_json_file('/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/dataset/AVIID/source/validation/',
                        '/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/dataset/AVIID/target/validation/',
                        '/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/dataset/AVIID/text/validation/',
                        '/data/hdd/sunhaisen/infrared/ControlNet/ControlNet-main/dataset/AVIID/text/validation_prompt.json')




