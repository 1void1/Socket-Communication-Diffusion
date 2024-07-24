
import socket
import torch
import cv2
import numpy as np
from models.irstd import IRSTD_UNET
from PIL import Image
from utils.config import *
import os
from utils.imagetotext import blip, imagetotext, generator_json_file
from utils.test import Test





def recv_all(sock, count):
    """Receiving all data from socket."""
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def process_image(model, filename,image_data, device):
    """Processing image using the model."""
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Failed to decode image")

    # 1.Send the image to the BLIP model to output prompt information
    target_image_path = split_and_save_image(image,filename)
    imagetotext(target_image_path,TEXT_PATH)
    # 2. Save the original image, target image and prompt information, and generate a json file
    source_path = os.path.join(SOURCE_FOLDER_SERVER,filename)
    target_path = os.path.join(TARGET_FOLDER_SERVER,filename)
    text_path = os.path.join(TEXT_PATH,filename.split('.')[0]+'.txt')
    output_json_path = generator_json_file(source_path,target_path,text_path,JSON_FILE_PATH)
    # 3. Generate pictures using ControlNet  json file    outputfile  prefix
    diffusion_model = Test()
    image_path = diffusion_model.generator_image_from_control_net(output_json_path,save_dir)
    image = cv2.imread(image_path)
    output_np = np.array(image)

    return output_np

def save_image(image_np, save_path):
    """Saving numpy image as a file."""
    im = Image.fromarray(image_np).convert('RGB')
    im.save(save_path)


def handle_client_connection(client_socket, model, device, save_dir):
    """Handle the client connection."""
    try:
        # 接收数据
        filename_length = int.from_bytes(recv_all(client_socket, 4), byteorder='big')
        filename = recv_all(client_socket, filename_length).decode()

        image_length = int.from_bytes(recv_all(client_socket, 4), byteorder='big')
        image_data = recv_all(client_socket, image_length)

        processed_image_np = process_image(model, filename,image_data, device)

        # Save the processed image to the server folder
        save_path = os.path.join(save_dir, filename)
        save_image(processed_image_np, save_path)

        # Encode and send the processed image back to the client
        _, processed_image_encoded = cv2.imencode('.png', processed_image_np)
        processed_image_data = processed_image_encoded.tobytes()

        client_socket.sendall(len(processed_image_data).to_bytes(4, byteorder='big'))
        client_socket.sendall(processed_image_data)
    except ValueError as e:
        print(f"Error: {e}")
        client_socket.sendall((0).to_bytes(4, byteorder='big'))
    finally:
        client_socket.close()








def start_server(model, device, save_dir, port=8002):
    """Start the server."""
    # 创建Socket套接字 监听端口
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('0.0.0.0', port))
    server_socket.listen()

    print(f"Server listening on port {port}...")
    while True:
        # 进行处理
        client_socket, client_address = server_socket.accept()
        print(f"Accepted connection from {client_address}")
        handle_client_connection(client_socket, model, device, save_dir)





def split_and_save_image(image_np, filename):
    """
    Split a 256x512x3 numpy array into two 256x256x3 images and save them.

    Parameters:
    image_np (numpy.ndarray): The input image of shape 256x512x3.
    save_dir (str): The directory where the images will be saved.
    base_filename (str): The base filename for the saved images.
    """
    # 检查输入图像的形状是否为 256x512x3
    if image_np.shape != (256, 512, 3):
        raise ValueError("Input image must have shape 256x512x3")

    # 切割图像
    left_image = image_np[:, :256, :]
    right_image = image_np[:, 256:, :]

    # 构建保存路径
    left_save_path = os.path.join(SOURCE_FOLDER_SERVER,filename)
    right_save_path = os.path.join(TARGET_FOLDER_SERVER,filename)

    # 保存图像
    cv2.imwrite(left_save_path, left_image)
    cv2.imwrite(right_save_path, right_image)

    print(f"Images saved to {left_save_path} and {right_save_path}")

    return right_save_path




# if __name__ == "__main__":
#     # 保存结果的路径
#     os.makedirs(save_dir, exist_ok=True)
#     # 设置GPU设备
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 加载模型
#     model = load_model(model_dir, device)
#     # 处理服务
#     start_server(model, device, save_dir, port=SERVER_PORT)
