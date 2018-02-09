---
layout: post
title: Project Alethea
---


# Group Members:

1.  Amlaan Bhoi (abhoi)
2.  Somshubra Majumdar (smajum6)
3.  Debojit Kaushik
4.  Christopher Alphones

# Current Structure

The structure of the project directory has been determined. There will be two main branches to this repository : the actual codebase and the website management branch.

Codebase structure is hierarchical, structured in a way similar to top projects like Keras or Tensorflow to offer similar flexibility in code management.

- External : Contains sub-projects that are necessary as a means to extract either datasets by scraping, as a container area for preprocessing before being transferred to "data/raw", or as a container for holding large datasets.
  - External will hold many sub-parts of the project that will be required by the main part - "Staging", however will not be required at runtime of the entire system.
  - It is used as a source for either data or external code that will be used with no direct dependency on the "staging" area modules.

- Staging : Holds the core modules required by the system at runtime. Contains the several submodules that will cooperate in order to form the entire system and all of its functionality.
  - Staging holds 4 main core modules at the moment - "data", "ml", "preprocessing" and "utils"
    - Data : Holds the "raw" data before preprocessing, the preprocessed dataset in "datasets" and the trained models in "models"
    - ML: Will eventually contain the code for training and evaluating the ML subsystems that will interact with the datasets
    - Preprocessing: Will hold scripts that read in "raw" data and transform it into clean and useful "datasets".
    - Utils : A glue module that will hold general utility scripts used the ML, Preprocessing and other sub-modules that will be added as and when required.
  - Staging is used as the core backend for the frontend systems that will be designed and deployed later.

# Important Links

- [Course Project Repository](https://github.com/titu1994/IDS-Course-Project)
- [CS 418 Project Link](http://cs418.cs.uic.edu/project.html)
