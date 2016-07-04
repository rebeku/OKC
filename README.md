# OKC

In this repository is various code written to derive insight about the tastes, values, preferences and recreational activities of users of the online dating website OKCupid.  Specifically, we analyze a set of almost 60,000 profiles from the San Francisco Bay area that was released in 2012.

Feel free to visit INSERT URL HERE to view the Tableau Dashboard with some of the findings from this project.

Unfortunately, the .csv files are too large for me to post on github.  To reproduce the work for this project, take the following steps:

1) download the original dataset from the following url:
https://github.com/rudeboybert/JSE_OkCupid/blob/master/profiles.csv.zip

2) Run the Jupyter Notebook code/Split.ipynb to save training and test sets.

3) Run code/Feature_Processing.ipynb to perform some preliminary processing on essay texts.

4) Run faves.ipynb to analyze user essays and extract titles and counts of popular books, bands, movies, television shows, and food, and save them to a new .csv file.

5) Run code/Feature_importance_efficienter.ipynb to analyze word frequency for different demographic groups.

6) Convert Feature_importance_efficienter.ipynb to .py file using the following command
```jupyter nbconvert --to python Feature_importance_efficienter.ipynb```

7) Automatically generate code to analyze faves.csv by running code/faves_code_generation.ipynb

8) Run faves.py (created in step 7) in your terminal
```python faves.py```

9) You have now saved all files used to analyze data for the Tableau Dashboard.

10) I may be adding files for machine learning models to predict categorical information about users based on their essay text.  Please watch for updates.
