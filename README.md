# OKC

In this repository is various code written to derive insight about the tastes, values, preferences and recreational activities of users of the online dating website OKCupid.  Specifically, we analyze a set of almost 60,000 profiles from the San Francisco Bay area that was released in 2012.

Feel free to visit https://public.tableau.com/profile/rebecca.eve.cohen#!/vizhome/okc_final3_0/Main to view the Tableau Dashboard with some of the findings from this project.

Unfortunately, the .csv files are too large for me to post on github.  To reproduce the work for this project, take the following steps:

1) download the original dataset from the following url:
https://github.com/rudeboybert/JSE_OkCupid/blob/master/profiles.csv.zip

2) Run the Jupyter Notebook code/Split.ipynb to save training and test sets.

3) Run code/Feature_Processing.ipynb to perform some preliminary processing on essay texts.

4) View Optimizing_gender_classifier.ipynb to view my process for selecting and tuning models to classify the writer's gender based on essay text.  The strongest model acheived 95% accuracy during testing.

5) Stay tuned for machine learning models on other interesting features.

6) Run faves.ipynb to analyze user essays and extract titles and counts of popular books, bands, movies, television shows, and food, and save them to a new .csv file.

7) Run code/Feature_importance_efficienter.ipynb to analyze word frequency for different demographic groups.

8) Convert Feature_importance_efficienter.ipynb to .py file using the following command
```jupyter nbconvert --to python Feature_importance_efficienter.ipynb```

9) Automatically generate code to analyze faves.csv by running code/faves_code_generation.ipynb

10) Run faves.py (created in step 7) in your terminal
```python faves.py```

11) Run correlations_by_feature.ipynb to save some exploratory data analysis on the relationships between categoricals in the dataset

12) You have now saved all files used to analyze data for the Tableau Dashboard.

13) Run custom_word_finder.ipynb to analyze user essays and compute stats for those users who mention any word in the English language.

14) I may be adding files for machine learning models to predict categorical information about users based on their essay text.  Please watch for updates.
