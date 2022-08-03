# Automatic License Plate Recognition App

The project is developed using pytorch to detect the license plate of vehicles from Russia and uses custom OCR to recognize characters from the detected number. 

### The app includes
* service for inference
* database (Postgresql)
* database visualization dashboard (metabase)
* http server (nginx)

### Software Packs Needed
* Docker
* Docker-compose

### Service for inference
The detector is built and trained on the yolov5 model. The optical character recognition model uses pre-trained resnet18 for feature extraction and two bidirectional recursive layers

model architecture
![ocr_architecture](https://user-images.githubusercontent.com/85789260/182594343-e0a9a5cd-787c-461a-bbf1-43dd094e5f90.png)

Examples of how the model works

![M885MK196](https://user-images.githubusercontent.com/85789260/182595126-c3e429be-f5f7-41ad-98bd-fe49fdf4a3a3.jpg)
![M064PO96](https://user-images.githubusercontent.com/85789260/182595151-4f19f76c-5700-4403-9196-e4f58ffb464e.jpg)

### Postgresql
The database stores the data number and links to open the image through the browser

### Metabase
Dashboard for visualization simplifies work with the database, for quick number search
![image](https://user-images.githubusercontent.com/85789260/182598464-d74695c9-a90f-4d6a-81d5-58fa4b6c13f9.png)

### nginx
http server allows you to open images directly in the browser via a link from the dashboard
