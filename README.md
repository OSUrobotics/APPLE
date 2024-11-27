# Automated Picking Physical Load Exploration (APPLE) dataset

This repository contains code associated with the Automated Picking Physical Load Exploration (APPLE) dataset. The data is available on Zenodo: https://zenodo.org/records/14226920 

The APPLE dataset provides physical data related to robotic apple picking. The data was collected in the autumn of 2023 on three separate orchard plots.  We instrumented both the robot and the environment and collected synchronizable time series sensor data during robotic fruit picking. 

The accompanying software, in this repository consists of two classes, the applePick class the appleAnnotater class. The applePick class creates an object based on two Pandas dataframes. These dataframes are imported from the "AllInterpolatedData.csv" file and the "metadata.csv" file, respectively, using Pandas.read\_csv. This class serves several functions. The first is transforming data into useful formats and fetching it for the user. The class is also able to remove the initial retreat and approach portions of the picking attempt, leaving only data from the picking phase. Finally, there are several plotting functions to aid in the display of data.

The other class, appleAnnotater, is a tool that allows you to click on points in plotted data from the APPLE dataset and automatically record the time that is associated with that data. The user must specify a path to the APPLE dataset and a function that plots the data they wish to see. The tool will then automatically cycle through every picking attempt and display the data for annotation. Two example plotting functions are provided inside of a script that runs the annotation tool.
        
The user may additionally specify a list of data channels they would like to be recorded alongside the times. These channels should be specified as a list of strings. A wide range of strings is accepted for each channel. For example, the magnitude of the force on the wrist of the robot can be specified as "f", "f_mag", or "force". If the user wishes to guarantee that a given string will provide the expected data, they must check the options in the function export_data(). The user may then specify the number of samples they wish to be recorded before and after the selected time.

The dependencies for this repository can be installed using the provided requirements.txt file.
