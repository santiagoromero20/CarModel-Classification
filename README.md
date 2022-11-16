# Vehicle Image Classification

 The goal of this project is to train a model with vehicle images, which later on will be able to classify a given input (as image) as the respective vehicle. This was a large project with specific focus on the training of the model. 
 
 Why? 
 
 Well, because of the complexity of the Dataset, around 15k Vehicle Images. As you can imagine, to deal with such a huge Dataset effectively the need of a Cloud Service is indispensable.

![Alt ](img/white_bear.jpeg "Title")
 ## **Motivation, Technologies and Teachings**

 This project is so enriching, from whatever perspective you analyze it. It makes yourself getting familiar with a Cloud Servie, in my particular case **AWS Services**. Which were provided by my tutors on Anyone AI Bootcamp. Then, handling a Dataset with images. I have to learn new libraries to automate the splitting of the Dataset, to make an insightuful EDA and at the end to remove the background of the images. 
 
 Then, in consideration with the Model Training. The goal was to Fine-Tuned an already trained model (Resnet 50), to gain more knowledge on Vehicles in specific, plus all the stablished weights learned from more than 1 million Images when google trained it and release it to the community. In addition to this, working with Neural Networks, more specifically CNN. And being able to put all my theoretical knowledge into practice.

 Some examples of Frameworks and Libraries used **os**, **argparse** , **cv2**, **Detectron2**, **COCO**, **Tensorflow** and **Keras**.

 I hope you enjoy and learn from this project as much as I have!
 ## **Table of Contents**

 **[1. Vehicle Classification](#heading--1)**

  * [1.1. Download](#heading--1-1)
  * [1.2. Scripts](#heading--1-2)
    * [1.2.1. Prepare_train_test_dataset](#heading--2-1-1)
    * [1.2.2. Remove_background](#heading--2-1-1)
    * [1.2.3. Train](#heading--2-1-1)
  * [1.3. Notebooks](#heading--1-2)
    * [1.3.1. EDA](#heading--2-1-1)
    * [1.3.2. Evaluation](#heading--2-1-1)
  *  [1.4. Models](#heading--1-2)
     * [1.4.1. Resnet_50](#heading--2-1-1)
  *  [1.5. Utils](#heading--1-2)
     * [1.5.1. Data_aug](#heading--2-1-1)
     * [1.5.2. Detection](#heading--2-1-1)
     * [1.5.2. Utils](#heading--2-1-1)
  *  [1.6. Tests](#heading--1-2)
     * [1.6.1. Test_data_aug](#heading--2-1-1)
     * [1.6.2. Test_detection](#heading--2-1-1)
     * [1.6.3. Test_resnet_50](#heading--2-1-1)
     * [1.6.4. Test_utils](#heading--2-1-1)
* [1.7. Docker](#heading--2-1-1)
     * [1.7.1. Dockerfile](#heading--2-1-1)
     * [1.7.1. Dockerfile_aws](#heading--2-1-1)
* [1.8. Experiments](#heading--2-1-1)
     * [1.8.1. Config_exp](#heading--2-1-1)

`1.1. download.py` : This file was coded to download the Dataset with the Vehicle Images and the CSV file with its corresponding labels from the s3 Bucket from AWS.

`1.2. Scripts/` : This folder contains various important scripts use on different parts of the project. Each one of them has a clear description of its role and functionallity.

`1.3. Notebooks/` : This folder contains two Jupyter Notebooks, the EDA and the Evaluation which basically is the evaluation notebook of the final selected model.

`1.4. Models/` : This folder contains a single script, which is in charge of the model creation. Here the CNN is coded with its respective hyperparameters.

`1.5. Utils/` : This folder contains various scripts which are use on different parts of the project. Same idea as the previously mentioned folder, Scripts. 

More particullary, the `utils/data_aug.py` script is called by the `models/resnet_50.py` at the time of creating the augmentation layers of the CNN.

Then, the `utils/utils.py` has several functions called at different stages of the project, while `utils/detection.py` is used when removing the noise from the images background.

`1.6. Tests/` : As the name indicates, there is not a lot of mistery with this file. It contains tests for 
different parts of the project.

`1.7. Docker/` : Also, little to say about this. It contains two Dockerfiles, one use for my local enviroment and the other one when I worked on the AWS Cloud

`1.8. Experiments/` : This folder contains an example of a YAML file, which is the input of the `scripts/train.main(config_file)` function, in charge of the training of the CNN. Every time you want to train a new model, you will have to create a new sub-folder inside `experiments/`.

**I really recommend reading the `VehicleClassWorkflow.docx` to gain a deeper and clearer understanding**
## **Installation**

You can use `Docker` to easily install all the needed packages and libraries. Two Dockerfiles are provided for both CPU and GPU support.

- **CPU:**

```bash
$ docker build -t s04_project -f docker/Dockerfile .
```

- **GPU:**

```bash
$ docker build -t s04_project -f docker/Dockerfile_gpu .
```

### Run Docker

```bash
$ docker run --rm --net host --gpus all -it \
    -v $(pwd):/home/app/src \
    --workdir /home/app/src \
    s04_project \
    bash
```

### Run Unit test

```bash
$ pytest tests/
```