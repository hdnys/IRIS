# SMART PLD Template Repository

![Insalogo](./images/logo-insa_0.png)

Template by [Riccardo Tommasini](riccardotommasini.com/) from [INSA Lyon](https://www.insa-lyon.fr/).

Students: **[To be assigned]**

### Abstract

## Description 

## Project Objectives

## Requirements

## How to Run the Project

## Checklist

- [ ] Clone the created repository offline;
- [ ]  Add your name and surname to the Readme file and your teammates as collaborators
- [ ] Complete the field above after the project is approved
- [ ]  Make any changes to your repository according to the specific assignment;
- [ ]  Ensure code reproducibility and instructions on how to replicate the results;
- [ ]  Add an open-source license, e.g., Apache 2.0;



## Description of the project 

Iris is a software first hardware agnostic tool for people with vision defects, the goal is to take advantage of local open source artificial intelligence models and more specific tools for facial recognition, object deteciton, detection of emotions, ect... The architecture should be as agnostic to the models as possible. The video stream will be received as frames by an orchestrator, this orchestrator will first receive frames from the video, after running lightweight similarity checks with the last frame received, if the frame is deemed different enough it will be passed on to the rest of the program. The program follows a common data pool archetecture where a non relational database, a json with static fields and dynamic fields is modified by each model (with interfaces for simpler models) to build a coherent context. The pipeline is the following, a VLM receives the frame and access the old json context, it either modifies the context or creates a new one depending on how different it is. Then based on the new static fields the orchestrator calls on the respective models such as object detection and categorization, facial recognition. Each of of the models outputs a json that we then use to augment the context. After finalizing the context, that same VLM now acting as an LLM takes in the context and types out a description to the user based on the context and a system prompt that distinguishes different types of blindness.  