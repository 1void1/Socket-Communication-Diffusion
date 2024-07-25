# Socket-Communication-Diffusion
## Project Description
This project is based on the Socket communication of the diffusion model
1. The client stores the dual-modal data set of infrared and visible light pairs;
2. The dual-modal data is spliced ​​and sent to the server;
3. The server listens to the data, splits the image and uses BLIP (Bootstrapping Language-Image Pre-training) to generate prompt information for the target image;
4. Generate a JSON file with the prompt information and the infrared and visible light dual-modal data;
5. Send it to the ControlNet pre-trained model to generate an infrared image;
6. The generated image is transmitted from the server to the client;
## Train with Your Own Data
1. Prepare a dual-modal dataset and store it in the source (original image) and target (target image) directories;
2. The server calls server.py to listen;
3. The client calls client.py to splice and send data to the specified IP and port;
   
Note: You need to modify the data storage path in the config.py file as well as the IP address and port of the client and server

## Result
The left side is the generated infrared image, the middle is the visible light image (source), and the right side is the infrared image (target)  

![Generate image](https://github.com/1void1/Socket-Communication-Diffusion/blob/main/result/03983.png)
![Source_image](https://github.com/1void1/Socket-Communication-Diffusion/blob/main/source/03983.png)
![Target_image](https://github.com/1void1/Socket-Communication-Diffusion/blob/main/target/03983.png)

![Generate image](https://github.com/1void1/Socket-Communication-Diffusion/blob/main/result/04129.png)
![Source_image](https://github.com/1void1/Socket-Communication-Diffusion/blob/main/source/04129.png)
![Target_image](https://github.com/1void1/Socket-Communication-Diffusion/blob/main/target/04129.png)

## Related Resources
Github: https://github.com/lllyasviel/ControlNet  
Github: https://github.com/salesforce/BLIP/  

