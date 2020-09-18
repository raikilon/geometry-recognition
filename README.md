# Automated recognition of features in architectural 3D models, using Artificial Intelligence Methods

Building Information Modelling (BIM) is a ground-breaking new approach to digitise the entire process of planning, designing, and constructing in the fields of Architecture, Engineering, and Construction (AEC), allowing interdisciplinary teams to collaborate on one single dataset. BIM data include all kinds of meta-information, ranging from the general type of each part of the model to its ordering number. While the adoption of BIM is steadily growing, it will still take another 5-10 years before most projects will fully use BIM instead of the traditional ComputerAided Design (CAD) approach. During this period of transition, and also after, CAD projects are not fully compatible with modern tools, as they lack crucial meta-information about the parts of the 3D models.

One of these advanced tools is [HEGIAS](https://www.hegias.com) , a Content-Management System for Virtual Reality (VRCMS). It requires certain meta-information in architectural 3D models to offer a fully automated CAD import and to provide an intuitive experience to its users, who explore and modify unbuilt homes and shops in immersive Virtual Reality, in single and multi-user.

The most critical meta-data is the class of the object. This information allows clustering similar CAD models together to create a catalogue and to get valuable information about the scene. An example could be the identification of walls which limit the movement of a user and can be used to append other objects such as shelves and paintings

## Objectives

The main objective of this Master’s Thesis is to research and find the most suitable machine learning model for 3D object classification. The second step is to develop a proof of concept of an automated workflow that recognises features in traditional CAD models and enhances them with essential metadata. More precisely, given a single object or a scene with multiple objects as input, the system should extract and classify each model as one of the following categories: floor, wall, stair and furniture (the more classes, the better). 

The current HEGIAS’ physics engine uses cuboids to represents the boundaries of models and to compute collisions. To be able to import the CAD models directly into the system and use them, the physics engine needs their simplified representation. For this reason, the proof of concept should contain a post-processing step which generates a tight and precise cuboid or a hierarchy of them for each input model.

## Installation

This project requires [Python](https://www.python.org/) 3.7.4+.

To install all the dependencies run the following command

```
pip install -r requirements.txt
```

## Credits

The project was realized by **Noli Manzoni** (nolimanzoni94@gmail.com) for the Master' Thesis at the [Università della Svizzera italiana](https://www.usi.ch).