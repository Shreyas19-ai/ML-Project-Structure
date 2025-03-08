## This is an End to End ML Project Structure Implementation

- Here we are not reading the data from the db.

## DVC
- Track our data files through DVC.
- Rest other files are tracked through git.
- So we have to make sure that our data file/folder/directory is not tracked by git.

> How to track through DVC
```
dvc add your_folder/file_name
```
- This will create 2 files 
1. file_name.dvc
2. .gitignore -- inside which will be your file_name, it means git will ignore your data file.
3. If we make any changes into the data then again we have to do " dvc add    " to track those changes.

## What we will track through git then?
- We will track file_name.dvc and .gitignore through git