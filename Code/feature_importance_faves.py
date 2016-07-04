# remove beginning of code prior to '# Save any features we might want to filter by to tf dataframe'
# replace with beginning

code = beginning + code.split('# Save any features we might want to filter by to tf dataframe', 1)[1]

import re

# Beginning of code is significantly different
# Rewrite manually

beginning = ("

import pandas as pd
import numpy as np
from time import time
import re

okc = pd.read_csv('../Assets/A/one_long_essay.csv', index_col='Unnamed: 0')
tf = pd.read_csv('../Assets/A/faves.csv', index_col='Unnamed: 0')

### Drop problematic/questionable values from faves
# Return to this section and expand as you identify new "favorites" that are most likely regex false positives in most cases.

# Most references to "love" are not the 2015 3-d erotic film, but more likely to be "I love Harry Potter!"
# "Chocolat" will be selected for those people whose favorite food is chocolate as well as those who enjoy th 2001 French film
# Similarly, 'it', '...', 'yes', '2', 'i', 'the', 'currently' 'and', 'oh', 'etc', 'big' and 'eat', 'tron', and 'pi' are most likely not titles or even necessarily whole words.
# 'Kurt Vonnegut' will select for a subset of 'Vonnegut' mentions.  Does not contribute new insight.
# 'elf' will be picked up by any mention of 'self'
# 'fiction' will pick up all mentions of non-fiction
# 'up' is most likely not the amazing animated movie :-(
# can I improve my regex so it only searches for whole words?
# TRY THAT LATER!

tf = tf.drop(['love', 'chocolat', 'it', '...', 'yes', '2', 'i', 'the', 'currently', 'and', 'oh', 'etc', 'big', 'eat', 'tron', 'the hangover', 'pi', 'kurt vonnegut', 'elf', 'fiction', 'up'], axis=1)
length = tf.shape[1]

")

# Copy and paste all code from Feature_importance_efficienter.ipynb
# Try again later using cog for style points (plus to capture any updates to feature_importance_efficienter with each running)
# For now I will skip commands of the form "men_list" that will run in ipynb but not in a .py file
# ultimately I will remove these commands from notebook and save all outputs to .txt and .csv files only

code = '''
import pandas as pd
import numpy as np
from time import time

okc = pd.read_csv('../Assets/A/one_long_essay.csv', index_col='Unnamed: 0')

def denull(essay):
    if type(essay) == float:
        return ''
    else: return essay

okc.essays = okc.essays.apply(denull)

from sklearn.feature_extraction.text import TfidfVectorizer

t0 = time()
vec = TfidfVectorizer(encoding='utf-8', stop_words='english', max_features=2000)
tf = vec.fit_transform(okc.essays)
print "vectorized essays in %g seconds" %(time()-t0)

tf = pd.DataFrame(tf.toarray(), columns=vec.get_feature_names())

# Save any features we might want to filter by to tf dataframe
features = ['age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'ethnicity',
           'height', 'income', 'job', 'offspring', 'orientation', 'pets', 'religion',
            'sex', 'sign', 'smokes', 'speaks']

for feature in features:
    tf['X%s' %feature] = okc[feature]

mean_words = tf.iloc[:,:2000].mean(axis=0)
mean_words.sort_values(inplace=True, ascending=False)

# add something to this for hue?
mean_words[:60].to_csv('../Assets/Tableau/top_words.csv')

# Compute mean value of each of first 2000 columns for men
t0 = time()
mdf = tf[tf.Xsex=='m']
df = pd.DataFrame(mdf.iloc[:,:2000].mean(axis=0), columns = ['m'])
print time()-t0

# Compute mean value of each of first 2000 columns for women
t0 = time()
fdf = tf[tf.Xsex == 'f']
df['f'] = fdf.iloc[:,:2000].mean(axis=0)
print time()-t0

df['diff'] = df['m']-df['f']

df.sort_values('diff', inplace=True)

# Save top 200 words for women and men for report
women_list = ''
for word in df.index[:200]:
    women_list = women_list + word + ', '

men_list = ''
# Write men_list in reverse order so most popular words are on top
for i in range(-1, -200, -1):
    men_list = men_list + df.index[i] + ', '

# order top females words by 'f' column
df_f = df.head(10)
df_f = df_f.sort_values('f', ascending=False)

# order top male words by 'm' column
df_m = df.tail(10)
df_m = df_m.sort_values('m', ascending=True)

# Create new df with top 10 features for men and top 10 features for women

df_mf = df_f.append(df_m)
df_mf.to_csv('../Assets/Tableau/df_mf.csv')

# Compare word frequency for social drinkers vs. heavy drinkers vs. non drinkers

# Compute mean value of each of first 2000 columns for social drinks
t0 = time()
ddf = tf[tf.Xdrinks=='socially']
df = pd.DataFrame(ddf.iloc[:,:2000].mean(axis=0), columns = ['social'])
print time()-t0

# Compute mean value of each of first 2000 columns for non-drinkers
t0 = time()
ddf = tf.query("Xdrinks == 'rarely'|Xdrinks == 'not at all'")
df['non-drinker'] = ddf.iloc[:,:2000].mean(axis=0)
print time()-t0

# Compute mean value of each of first 2000 columns for heavy drinkers
t0 = time()
ddf = tf.query("Xdrinks == 'often'|Xdrinks == 'very often'|Xdrinks == 'desperately'")
df['heavy'] = ddf.iloc[:,:2000].mean(axis=0)
print time()-t0

df['avg'] = df.mean(axis=1)
# Compute differences between social drinkers' values and other categories
df['nd_diff'] = df['non-drinker'] - df['avg']
df['h_diff'] = df['heavy'] - df['avg']
df['s_diff'] = df['social'] - df['avg']

# Save top 200 words for heavy, social, and non-drinkers for report
# heavy drinkers:
df.sort_values('h_diff', ascending=False, inplace=True)

heavy_list = ''
for word in df.index[:200]:
    heavy_list = heavy_list + word + ', '

# social drinkers:
df.sort_values('s_diff', ascending=False, inplace=True)

social_list = ''
for word in df.index[:200]:
    social_list = social_list + word + ', '

# non-drinkers:
df.sort_values('nd_diff', ascending=False, inplace=True)

non_list = ''
for word in df.index[:200]:
    non_list = non_list + word + ', '

# For categories with more than two labels, produce packed bubbles instead of bar charts

# Save top 18 from each cat
# non-drinkers first
df.sort_values('nd_diff', inplace=True, ascending=False)
nd = pd.DataFrame(df.head(18)['non-drinker'])
nd.columns = ['tfidf']
nd['label'] = 'non-drinker'

# social drinkers
df.sort_values('s_diff', inplace=True, ascending=False)
sd = pd.DataFrame(df.head(18)['social'])
sd.columns = ['tfidf']
sd['label'] = 'social'

# heavy drinkers
df.sort_values('h_diff', inplace=True, ascending=False)
hd = pd.DataFrame(df.head(18)['heavy'])
hd.columns = ['tfidf']
hd['label'] = 'heavy'

df_drinks = pd.concat([nd, sd, hd], axis=0)

df_drinks.to_csv('../Assets/Tableau/df_drinks.csv')

from matplotlib import pyplot as plt
%matplotlib inline

plt.hist(okc.age, 50, normed=1, facecolor='green', alpha=0.75)
plt.axis([15, 80, 0, .1])
plt.show

# Compare under 30 to over 30

def age_encoder(age):
    if age ==0:
        return np.nan
    elif age < 30:
        return 0
    else:
        return 1

tf.Xage = tf.Xage.apply(age_encoder)

# Compare word frequency for under 30 vs. over 30

# Compute mean value of each of first 2000 columns for over 30
t0 = time()
odf = tf[tf.Xage==1]
df = pd.DataFrame(odf.iloc[:,:2000].mean(axis=0), columns = ['>30'])
print time()-t0

# Compute mean value of each of first 2000 columns for under 30
ydf = tf[tf.Xage == 0]
df['<30'] = ydf.iloc[:,:2000].mean(axis=0)
print time()-t0

df['diff'] = df['>30']-df['<30']
df.sort_values('diff', inplace=True)

# Save top 200 words for people under and over 30
# older people:
df.sort_values('diff', ascending=False, inplace=True)

older_list = ''
for word in df.index[:200]:
    older_list = older_list + word + ', '

# younger people
younger_list = ''
# Write older_list in reverse order so most popular words are on top
for i in range(-1, -200, -1):
    younger_list = younger_list + df.index[i] + ', '

# Create new df with top 10 features for <30 and top >30

df_y = df.head(10)
df_y = df_y.sort_values('<30', ascending=False)

df_o = df.tail(10)
df_o = df_o.sort_values('>30')

df_age = pd.concat([df_y,df_o])
df_age.to_csv('../Assets/Tableau/df_age.csv')

def drug_encoder(drugs):
    if drugs == 'never':
        return 0
    elif drugs == 'sometimes' or drugs == 'often':
        return 1
    else:
        return np.nan

tf.Xdrugs = tf.Xdrugs.apply(drug_encoder)

# Compute mean value of each of first 2000 columns for drug users
t0 = time()
udf = tf[tf.Xdrugs==1]
df = pd.DataFrame(odf.iloc[:,:2000].mean(axis=0), columns = ['users'])
print time()-t0

# Compute mean value of each of first 2000 columns for non-users
ydf = tf[tf.Xdrugs == 0]
df['non-users'] = ydf.iloc[:,:2000].mean(axis=0)
print time()-t0

df['diff'] = df['users']-df['non-users']
df.sort_values('diff', inplace=True)

# Save top 200 words for users and non-users
# users:
df.sort_values('diff', ascending=False, inplace=True)

non_user_list = ''
for word in df.index[:200]:
    non_user_list = non_user_list + word + ', '

# non-users
user_list = ''
# Write older_list in reverse order so most popular words are on top
for i in range(-1, -200, -1):
    user_list = user_list + df.index[i] + ', '

# Create new df with top 10 features for users and nonusers

df_users = df.head(10)
df_users = df_users.sort_values('users', ascending=False)

df_non = df.tail(10)
df_non = df_non.sort_values('non-users')

df_drugs = pd.concat([df_users, df_non], axis=0)
df_drugs.to_csv('../Assets/Tableau/df_drugs.csv')

plt.hist(tf.Xincome[tf.Xincome != -1], 50, normed=1, facecolor='green', alpha=0.75)
plt.show

# Divide income income into <50k, 50-100k, >100k

def income_encoder(income):
    if income == -1:
        return -1
    elif income <= 50000:
        return 0
    elif income <= 100000:
        return 1
    else:
        return 2

tf.Xincome = tf.Xincome.apply(income_encoder)

# Compare word frequency for different income levels

# Compute mean value of each of first 2000 columns for lower income
t0 = time()
pdf = tf[tf.Xincome==0]
df = pd.DataFrame(pdf.iloc[:,:2000].mean(axis=0), columns = ['<50k'])
print time()-t0

# Compute mean value of each of first 2000 columns for middle income
t0 = time()
mdf = tf[tf.Xincome==1]
df['50-100k'] = mdf.iloc[:,:2000].mean(axis=0)
print time()-t0

# Compute mean value of each of first 2000 columns for highest income
t0 = time()
rdf = tf[tf.Xincome==2]
df['>100k'] = rdf.iloc[:,:2000].mean(axis=0)
print time()-t0


# Compute differences between each income category and overall average
df['avg'] = df.mean(axis=1)
df['r_diff'] = df['>100k'] - df['avg']
df['p_diff'] = df['<50k'] - df['avg']
df['m_diff'] = df['50-100k'] - df['avg']

# Save lists for each income bracket

df.sort_values('r_diff', ascending=False, inplace=True)

rich_list = ''
for word in df.index[:200]:
    rich_list = rich_list + word + ', '

df.sort_values('m_diff', ascending=False, inplace=True)

mid_list = ''
for word in df.index[:200]:
    mid_list = mid_list + word + ', '


df.sort_values('p_diff', ascending=False, inplace=True)
lower_list = ''
for word in df.index[:200]:
    lower_list = lower_list + word + ', '

# For categories with more than two labels, produce packed bubbles instead of bar charts

# Save top 18 from each cat
# richest first
df.sort_values('r_diff', inplace=True, ascending=False)
df2 = pd.DataFrame(df.head(18)['>100k'])
df2.columns = ['tfidf']
df2['label'] = '> 100k'

# middle income
df.sort_values('m_diff', inplace=True, ascending=False)
df3 = pd.DataFrame(df.head(18)['50-100k'])
df3.columns = ['tfidf']
df3['label'] = '50 - 100k'

# lower income
df.sort_values('p_diff', inplace=True, ascending=False)
df4 = pd.DataFrame(df.head(18)['<50k'])
df4.columns = ['tfidf']
df4['label'] = '< 50k'

df_income = pd.concat([df2, df3, df4], axis=0)
df_income.to_csv('../Assets/Tableau/df_income.csv')

# calculate dummy columns for likes dogs and likes cats
# will be some overlap since not mutually exclusive

def mask_dogs(pets):
    # catches 'has dogs' or 'likes dogs'
    try:
        if pets.find('dogs') > -1:
            if pets.find('dislikes dogs') == -1:
                return 1
        else:
            return 0
    except:
        return 0

def mask_cats(pets):
    try:
        if pets.find('cats') > -1:
            if pets.find('dislikes cats') == -1:
                return 1
        else:
            return 0
    except:
        return 0

tf['Xdogs'] = tf.Xpets.apply(mask_dogs)
tf['Xcats'] = tf.Xpets.apply(mask_cats)

# find mean scores for dog people

t0 = time()
ddf = tf[tf.Xdogs==1]
df = pd.DataFrame(ddf.iloc[:,:2000].mean(axis=0), columns = ['dogs'])
print time()-t0

# find mean scores for cat people
cdf = tf[tf.Xcats == 1]
df['cats'] = cdf.iloc[:,:2000].mean(axis=0)
print time()-t0

df['diff'] = df['dogs'] - df['cats']
df.sort_values('diff', inplace=True)

# remove cat, cats, dog and dogs from this list since they don't contribute insight
df = df.drop(['cat', 'cats', 'dog', 'dogs'], axis=0)

# Save top 200 words for cats and dogs for report
cat_list = ''
for word in df.index[:200]:
    cat_list = cat_list + word + ', '

dog_list = ''
# Write dog_list in reverse order so most popular words are on top
for i in range(-1, -200, -1):
    dog_list = dog_list + df.index[i] + ', '

# Sort top 10 values for cats
df_cats = df.head(10).sort_values('cats')

# Sort top 10 values for dogs
df_dogs = df.tail(10).sort_values('dogs', ascending=False)

# Save df pets
df_pets = pd.concat([df_dogs, df_cats], axis=0)
df_pets.to_csv('../Assets/Tableau/df_ed.csv')

def ed_encoder(ed):
    # a person is a college grad if they either "graduated from college/university"
    # or mention law school, med, school, masters program or ph. d program (all instances of the word program are graduate )
    try:
        if ed == 'graduated from college/university' or ed.find('law') >= 0 or ed.find('med') >= 0 or ed.find('program') >= 0:
            return 1
        # space camp answers are facetious and must be excluded
        # BTW I am in space camp right now
        elif ed.find('space camp') >= 0:
            return np.nan
        else: return 0
    except:
        return np.nan

tf.Xeducation = tf.Xeducation.apply(ed_encoder)

# Compare word frequency for college educated vs. not college

# Compute mean value of each of first 2000 columns for college
t0 = time()
df2 = tf[tf.Xage==1]
df = pd.DataFrame(df2.iloc[:,:2000].mean(axis=0), columns = ['grad'])
print time()-t0

# Compute mean value of each of first 2000 columns for under 30
df2 = tf[tf.Xage == 0]
df['non-grad'] = df2.iloc[:,:2000].mean(axis=0)
print time()-t0

df['diff'] = df['grad'] - df['non-grad']
df.sort_values('diff', inplace=True)

# Save top 200 words for grads and non-grads for report
non_grad_list = ''
for word in df.index[:200]:
    non_grad_list = non_grad_list + word + ', '

grad_list = ''
# Write grad_list in reverse order so most popular words are on top
for i in range(-1, -200, -1):
    grad_list = grad_list + df.index[i] + ', '

# Sort top 10 words for non-grad
df3 = df.head(10).sort_values('non-grad', ascending=False)

# Sort top 10 words for grad
df4 = df.tail(10).sort_values('grad')

df = pd.concat([df3, df4], axis=0)

df.to_csv('../Assets/Tableau/df_ed.csv')

# It will be more efficient to encode dummies for each ethnic group than to search each entry
# People who list multiple ethnicities will count for all ethnicities listed

# write encoder for each ethnicity

groups = ['asian', 'middle eastern', 'black', 'native american', 'indian', 'pacific islander', 'hispanic / latin', 'white']

for ethnicity in groups:
    def ethnicity_encoder(eth):
        global ethnicity
        try:
            if eth.find(ethnicity) >= 0:
                return 1
            else: return 0
        except:
            return 0

    tf['X%s' %ethnicity] = tf.Xethnicity.apply(ethnicity_encoder)

df=pd.DataFrame(index=tf.columns[:2000])
# Compute mean value of each of first 2000 columns for each ethnic group
for ethnicity in groups:
    t0 = time()
    df2 = tf[tf['X%s' %ethnicity] == 1]
    df[ethnicity] = pd.DataFrame(df2.iloc[:,:2000].mean(axis=0))
    print time()-t0

df['avg'] = df.mean(axis=1)

for ethnicity in groups:
    df['%s_diff' %ethnicity] = df[ethnicity] - df['avg']

# List top 200 words for each ethnicity

# Create an empty string assigned to each ethnicity
eth_dic = {}
for ethnicity in groups:
    eth_dic[ethnicity] = ''

# Fill string for each ethnicity with top words
for ethnicity in groups:
    df.sort_values('%s_diff' %ethnicity, ascending=False, inplace=True)

    for word in df.index[:200]:
        eth_dic[ethnicity] = eth_dic[ethnicity] + word + ', '

# There are 8 different ethnic groups available
# Tableau does well with about 50 to 60 words for packed bubbles
# Take top 7 words for each ethnicity

# Drop words asian, middle, eastern, indian, and india as they are top ranking words but redundant with categories

df.drop(['asian', 'middle', 'eastern', 'indian', 'india'], axis=0, inplace=True)

# must have at least one row in dataframe in order to use pd.concat
df_ethnicity = pd.DataFrame(index=[0], columns=['tfidf', 'label'])

for ethnicity in groups:
    # Save top 7 words for each ethnicity
    df.sort_values('%s_diff' %ethnicity, ascending=False, inplace=True)
    df3 = pd.DataFrame(df.head(7)[ethnicity])
    df3.columns=['tfidf']
    df3['label'] = ethnicity

    df_ethnicity = pd.concat([df_ethnicity, df3], axis=0)

# drop dummy row from top
df_ethnicity.drop(0, axis=0, inplace=True)

df_ethnicity.to_csv('../Assets/Tableau/df_ethnicity.csv')


df = pd.DataFrame(index=tf.columns[:2000])
df_jobs = pd.DataFrame(index=[0], columns=['tfidf', 'label'])
job_dic = {}
# Most popular job category is "other"  Exclude that
for job in tf.Xjob.value_counts()[1:11].index:
    # Save a column of dummies for each job category
    def job_encoder(career):
        global job
        try:
            if career.find(job) >= 0:
                return 1
            else: return 0
        except:
            return 0

    tf['X%s' %job] = tf.Xjob.apply(job_encoder)

    # Compute mean tfidf scores for job type
    t0 = time()
    df2 = tf[tf['X%s' %job] == 1]
    df[job] = pd.DataFrame(df2.iloc[:,:2000].mean(axis=0))
    print time()-t0

# Compute overall mean tfidf scores
df['avg'] = df.mean(axis=1)


for job in tf.Xjob.value_counts()[1:11].index:
    # compute diff between scores for each job and overall mean
    df['%s_diff' %job] = df[job] - df['avg']


    # sort by job
    # save top 200 words as dictionary values for each job
    # save top 6 words with tfidf scores and labels to df_jobs
    job_dic[job] = ''
    df.sort_values('%s_diff' %job, ascending=False, inplace=True)
    # save top 200 words to reserved string in job_dic
    for word in df.index[:200]:
        job_dic[job] = job_dic[job] + word + ', '

    # Save top 6 words to df_jobs
    df.sort_values('%s_diff' %job, ascending=False, inplace=True)

    df3 = pd.DataFrame(df.head(6)[job])
    df3.columns=['tfidf']
    df3['label'] = job

    df_jobs = pd.concat([df_jobs, df3], axis=0)

# drop dummy row from top
df_jobs.drop(0, axis=0, inplace=True)

df_jobs.to_csv('../Assets/Tableau/df_jobs.csv')

# Compute mean value of each of first 2000 columns for gay men

df2 = tf[tf.Xsex=='m'][tf.Xorientation=='gay']
df = pd.DataFrame(df2.iloc[:,:2000].mean(axis=0), columns = ['gay men'])


# Compute mean value of each of first 2000 columns for bi men
df2 = tf[tf.Xsex=='m'][tf.Xorientation=='bisexual']
df['bi men'] = df2.iloc[:,:2000].mean(axis=0)


# Compute mean values for straight men
df2 = tf[tf.Xsex=='m'][tf.Xorientation=='straight']
df['straight men'] = df2.iloc[:,:2000].mean(axis=0)


# Compute mean values for gay women
df2 = tf[tf.Xsex=='f'][tf.Xorientation=='gay']
df['gay women'] = df2.iloc[:,:2000].mean(axis=0)

# Compute mean values for bi women
df2 = tf[tf.Xsex=='f'][tf.Xorientation=='bisexual']
df['bi women'] = df2.iloc[:,:2000].mean(axis=0)

# Compute mean values for straight women
df2 = tf[tf.Xsex=='f'][tf.Xorientation=='straight']
df['straight women'] = df2.iloc[:,:2000].mean(axis=0)


df['avg'] = df.mean(axis=1)
df['gay men diff'] = df['gay men'] - df.avg
df['bi men diff'] = df['bi men'] - df.avg
df['straight men diff'] = df['straight men'] - df.avg
df['gay women diff'] = df['gay women'] - df.avg
df['bi women diff'] = df['bi women'] - df.avg
df['straight women diff'] = df['straight women'] - df.avg

# Save word lists
df_orientation = pd.DataFrame(index=[0], columns=['tfidf', 'label'])
group_dic = {}
for group in df.columns[:6]:
    df.sort_values('%s diff' %group, ascending=False, inplace=True)
    group_dic[group] = ''
    for word in df.index[:200]:
        group_dic[group] = group_dic[group] + word + ', '

# Save df for Tableau with top 10 for each category
    df2 = pd.DataFrame(df[group].head(10))
    df2.columns = ['tfidf']
    df2['label'] = group


    df_orientation = pd.concat([df_orientation, df2], axis=0)

df_orientation.drop(0, axis=0, inplace=True)

df_orientation.to_csv('../Assets/Tableau/df_orientation.csv')

# list religions
religions = []
for religion in tf.Xreligion.value_counts().index:
    rel = religion.split(' ', 1)[0]
    if rel not in religions:
        religions.append(rel)
religions = filter(lambda x: x != 'other', religions)

# Encode dummies for each religion
# rel is individual person's religion string "Christianity and very serious about it"
# religion is each group "Christianity"

tf.Xreligion.replace(np.nan, '', inplace=True)

df = pd.DataFrame()
rel_dic = {}
df_religion = pd.DataFrame(index=[0], columns=['tfidf', 'label'])

for religion in religions:
    tf['X%s' %religion] = tf.Xreligion.apply(lambda rel: 1 if rel.split(' ', 1)[0] == religion else 0)

    # Compute mean values for each religion
    df2 = tf[tf['X%s' %religion]==1]
    df[religion] = df2.iloc[:,:2000].mean(axis=0)

# Compute mean for each word across religons
df['avg'] = df.mean(axis=1)

for religion in religions:
    df['%s_diff' %religion] = df[religion] -df['avg']

    # Save top 200 words for each religion as string
    df.sort_values('%s_diff' % religion, inplace=True, ascending=False)

    rel_dic[religion] = ''
    for word in df.index[:200]:
        rel_dic[religion] = rel_dic[religion] + word + ', '

    # Save top 7 words for each religon as df for Tableau
    df2 = pd.DataFrame(df[religion].head(7))
    df2.columns = ['tfidf']
    df2['label'] = religion

    df_religion = pd.concat([df_religion, df2], axis=0)

df_religion.drop(0, axis=0, inplace=True)
df_religion.to_csv('../Assets/Tableau/df_religion.csv')


# Encode dummies for each religion
# rel is individual person's religion string "Christianity and very serious about it"
# religion is each group "Christianity"

tf.Xreligion.replace(np.nan, '', inplace=True)

df = pd.DataFrame()
rel_dic = {}
df_religion = pd.DataFrame(index=[0], columns=['tfidf', 'label'])

for religion in religions:
    tf['X%s' %religion] = tf.Xreligion.apply(lambda rel: 1 if rel.split(' ', 1)[0] == religion else 0)

    # Compute mean values for each religion
    df2 = tf[tf['X%s' %religion]==1]
    df[religion] = df2.iloc[:,:2000].mean(axis=0)

# Compute mean for each word across religons
df['avg'] = df.mean(axis=1)

for religion in religions:
    df['%s_diff' %religion] = df[religion] -df['avg']

    # Save top 200 words for each religion as string
    df.sort_values('%s_diff' % religion, inplace=True, ascending=False)

    rel_dic[religion] = ''
    for word in df.index[:200]:
        rel_dic[religion] = rel_dic[religion] + word + ', '

    # Save top 7 words for each religon as df for Tableau
    df2 = pd.DataFrame(df[religion].head(7))
    df2.columns = ['tfidf']
    df2['label'] = religion

    df_religion = pd.concat([df_religion, df2], axis=0)

df_religion.drop(0, axis=0, inplace=True)
df_religion.to_csv('../Assets/Tableau/df_religion.csv')

levels = []
for religion in tf.Xreligion.value_counts().index:
    try:
        level = religion.split('and ', 1)[1]
    except: continue
    if level not in levels:
        levels.append(level)

def level_encoder(rel):
    try:
        for level in levels:
            if rel.find(level) >0:
                return level
    except:
        return ''

tf['Xreligiousness'] = tf.Xreligion.apply(level_encoder)

df = pd.DataFrame()
df_religiousness = pd.DataFrame(index=[0], columns=['tfidf', 'label'])
level_dic = {}
for level in levels:
    # Compute mean values for each level of religiousness
    df2 = tf[tf['Xreligiousness']==level]
    df[level] = df2.iloc[:,:2000].mean(axis=0)
# Compute averages
df['avg'] = df.mean(axis=1)

for level in levels:
    # sort by diff for each category
    df['%s_diff' %level] = df[level] - df['avg']
    df.sort_values('%s_diff' % level, inplace=True, ascending=False)
    # Save top 200 to level_dic
    level_dic[level] = ''
    for word in df.index[:200]:
        level_dic[level] = level_dic[level] + word + ', '
    # Save top 20 to df
    df2 = pd.DataFrame(df[level].head(20))
    df2.columns = ['tfidf']
    df2['label'] = level

    df_religiousness = pd.concat([df_religiousness, df2], axis=0)
df_religiousness.drop(0, axis=0, inplace=True)

df_religiousness.to_csv('../Assets/Tableau/df_religiousness.csv')

tf.Xdiet.value_counts()

# ignore anything and other
# just look for vegetarian, vegan, kosher, halal

def diet_encoder(diet):
    try:
        if diet.find('vegan') >= 0:
            return 'vegan'
        elif diet.find('vegetarian') >= 0:
            return 'vegetarian'
        elif diet.find('kosher') >= 0:
            return 'kosher'
        elif diet.find('halal') >= 0:
            return 'halal'
        else:
            return ''
    except:
        return ''

tf.Xdiet = tf.Xdiet.apply(diet_encoder)

# Exclude '' from diets
diets = tf.Xdiet.value_counts().index[1:]

df = pd.DataFrame()
df_diet = pd.DataFrame(index=[0], columns=['tfidf', 'label'])
diet_dic = {}
for diet in diets:
    # Compute mean values for each level of religiousness
    df2 = tf[tf.Xdiet==diet]
    df[diet] = df2.iloc[:,:2000].mean(axis=0)
# Compute averages
df['avg'] = df.mean(axis=1)

for diet in diets:
    # sort by diff for each category
    df['%s_diff' %diet] = df[diet] - df['avg']
    df.sort_values('%s_diff' % diet, inplace=True, ascending=False)
    # Save top 200 to level_dic
    diet_dic[diet] = ''
    for word in df.index[:200]:
        diet_dic[diet] = diet_dic[diet] + word + ', '
    # Save top 20 to df
    df2 = pd.DataFrame(df[diet].head(15))
    df2.columns = ['tfidf']
    df2['label'] = diet

    df_diet = pd.concat([df_diet, df2], axis=0)
df_diet.drop(0, axis=0, inplace=True)

df_diet.to_csv('../Assets/Tableau/df_diet.csv')

# compare labels w/ encoders vs. labels with dummies for overall correlations
# write script to save dicts to text file

'''

# adjust indexing
# tf.iloc[:,:2000] adjusted to appropriate feature length (currently 1046 but may be adjusted if I drop more titles)
# save appropriate length as length
# tf.iloc[:,:2000] become tf.iloc[:,:length]

code = code.replace('2000', 'length')

# change top_words.csv to top_faves.csv
code = code.replace('top_words.csv', 'top_faves.csv')

# remove %matplotlib inline
code = code.replace('%matplotlib inline', '')

# change 'df' to 'faves' in FILENAMES ONLY to avoid saving over important files
# e.g. df_drinks.to_csv('../Assets/Tableau/df_drinks.csv') becomes df_drinks.to_csv('../Assets/Tableau/faves_drinks.csv')

# Look for any time df appears BETWEEN (Tableau/) and (.csv)
code = re.sub('Tableau\/df', 'Tableau/faves', code)

# remove all lines that include a .drop.  Rows being dropped from tf-idf will not exist in faves.

code = re.sub('\\n.*\.?drop.*?\\n', '', code)

# remove beginning of code prior to '# Save any features we might want to filter by to tf dataframe'
# replace with beginning

code = beginning + code.split('# Save any features we might want to filter by to tf dataframe', 1)[1]

f = open("test.py","w") #opens file with name of "test.py"

f.write(code)

f.close()
