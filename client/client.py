import socket
import cv2
import os
import numpy as np
from PIL.Image import Image

from utils.get_log import GetLogger
from utils.config import *

# read image
def read_image(image_path):
    '''
    reading image
    :param image_path:  input image path
    :return: Numpy type
    '''
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print_log.info(f"Reading the image: {image_path}")
    if image is None:
        print_log.error(f"Failed to load image {image_path}")
    return image

def get_image_files(folder):
    """
    get images
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

def is_image_file(file_path):
    """Check if a file is an image"""
    valid_image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    extension = os.path.splitext(file_path)[1].lower()

    # 检查扩展名是否为图片格式
    if extension not in valid_image_extensions:
        return False

    # 尝试用 OpenCV 打开文件
    try:
        img = cv2.imread(file_path)
        if img is not None:
            return True
        else:
            return False
    except Exception:
        return False


def contains_image(directory):
    """Check if the file contains an image"""
    for root,_,files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root,file)
            if is_image_file(file_path):
                return True
    return False

def concatenate_images(folder1, folder2, output_folder):
    """concate image"""
    files1 = get_image_files(folder1)
    files2 = get_image_files(folder2)

    for file1 in files1:
        filename = os.path.basename(file1)
        file2 = os.path.join(folder2, filename)

        if os.path.exists(file2):
            image1 = read_image(file1)
            image2 = read_image(file2)

            if image1 is not None and image2 is not None:

                if image1.shape == image2.shape:
                    concatenated_image = np.concatenate((image1, image2), axis=1)
                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, concatenated_image)
                    print(f"Saved concatenated image: {output_path}")
                else:
                    print(f"Image shapes are not the same: {file1}, {file2}")
        else:
            print(f"Corresponding file not found for {file1}")


def encode_image(image):
    """Image is encoded as byte data"""
    _, image_data = cv2.imencode('.png', image)
    print_log.info("Encoding image...")
    return image_data.tobytes()


def send_image_data(client_socket, filename, image_data):
    """Sening data to the server"""
    print_log.info("Sending image...")

    filename_encoded = filename.encode()
    client_socket.sendall(len(filename_encoded).to_bytes(4, byteorder='big'))
    client_socket.sendall(filename_encoded)

    client_socket.sendall(len(image_data).to_bytes(4, byteorder='big'))
    client_socket.sendall(image_data)
    print_log.info("Image data sent.")



def receive_processed_image(client_socket):
    """Receiving the processed image data"""
    processed_image_length = int.from_bytes(client_socket.recv(4), byteorder='big')
    if processed_image_length > 0:
        processed_image_data = b''
        while len(processed_image_data) < processed_image_length:
            processed_image_data += client_socket.recv(1024)
        processed_image = cv2.imdecode(np.frombuffer(processed_image_data, np.uint8), cv2.IMREAD_UNCHANGED)
        return processed_image
    else:
        print_log.error("Failed to process image")
        return None

# save image
def save_processed_image(processed_image, output_path):
    """Saving the processed image data"""
    cv2.imwrite(output_path, processed_image)
    print_log.info(f"Processed image saved as {output_path}")

def process_image(filename):
    """Processing individual images"""
    # get image path
    image_path = os.path.join(IMAGE_FOLDER, filename)
    # read image
    image = read_image(image_path)
    if image is None:
        return
    # encoding image to bytes
    image_data = encode_image(image)


    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((SERVER_IP, SERVER_PORT))
        # send image data  socket,image filename,image_data
        send_image_data(client_socket, filename, image_data)
        # receive image
        processed_image = receive_processed_image(client_socket)


        if processed_image is not None:
            os.makedirs(PROCESSED_FOLDER, exist_ok=True)
            output_path = os.path.join(PROCESSED_FOLDER, "Processed_" + filename)
            # save image
            save_processed_image(processed_image, output_path)


if __name__ == "__main__":
    print_log = GetLogger.get_logger()
    if not contains_image(IMAGE_FOLDER):
        concatenate_images(SOURCE_FOLDER,TARGET_FOLDER,IMAGE_FOLDER)
    for filename in os.listdir(IMAGE_FOLDER):
        if filename.endswith('.png'):
            process_image(filename)



