# Socket-Communication-Diffusion
## Project Description
This project is based on the Socket communication of the diffusion model
1. The client stores the dual-modal data set of infrared and visible light pairs;
2. The dual-modal data is spliced ​​and sent to the server;
3. The server listens to the data, splits the image and uses BLIP (Bootstrapping Language-Image Pre-training) to generate prompt information for the target image;
4. Generate a JSON file with the prompt information and the infrared and visible light dual-modal data;
5. Send it to the ControlNet pre-trained model to generate an infrared image;
6. The generated image is transmitted from the server to the client;
