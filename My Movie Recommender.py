#!/usr/bin/env python
# coding: utf-8

# # Brian's Movie Recommender

# While this is not a true recommendation system, it is still effective for film recommendations as it suggests films similiar to my choices.

# ## Libraries

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# ## Getting the Data

# In[3]:


columns_names = ['user_id','item_id','rating','timestamp']


# In[4]:


df = pd.read_csv('u.data', sep='\t',names=columns_names)


# In[5]:


df.head()


# In[6]:


movie_titles = pd.read_csv('Movie_Id_Titles')


# In[7]:


movie_titles.head()


# In[8]:


df = pd.merge(df,movie_titles,on='item_id')


# In[9]:


df.head()


# ## EDA

# ## Visualisation Imports

# In[10]:


import matplotlib.pyplot as plt


# In[11]:


import seaborn as sns


# In[12]:


sns.set_style('white')


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')


# I created a ratings dataframe with average rating and number of ratings:

# In[14]:


df.groupby('title')['rating'].mean().sort_values(ascending=False).head()


# These are the highest rated films in the dataset. It would appear only a few people have rated these as the films are very obscure. Sorting by the number of ratings:

# In[15]:


df.groupby('title')['rating'].count().sort_values(ascending=False).head()


# These films are a lot more well known so there is more variety in the ratings.

# In[16]:


ratings = pd.DataFrame(df.groupby('title')['rating'].mean())


# In[17]:


ratings.head()


# In[18]:


ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())


# In[19]:


ratings.head()


# In[20]:


ratings['num of ratings'].hist(bins = 70)


# From this histogram it is clear that most films in the dataset have under ten or so ratings.  Most people watch famous or blockbuster films so those films will have the most amount of ratings. There is a serious decline in the number of ratings after the 100 mark.

# In[21]:


ratings['rating'].hist(bins=70)


# There are peaks at the whole numbers 1, 2 , 3, 4 and 5. This makes sense as most people would give a film a whole number star. Most films are distributed normally around 3/3.5 stars. There are a lot of 1 star ratings because there is probably a lot of bad movies and people are more likely to give a film they disliked a 1 star review instead of a 2 star review.

# In[22]:


sns.jointplot(x='rating',y='num of ratings', data=ratings,alpha=0.5)


# From this jointplot, it seems that the more ratings a film gets, the higher the rating is likely to be. So the higher rating a film gets, the more likely it is to be seen by more people and will receive even more ratings.

# ## Recommending Similar Movies

# Here I use a pivot table to create a matrix that has the user ids on one axis and the movie title on another axis. Each cell will then consist of the rating the user gave to that movie. There are a lot of NaN values, because most people have not seen most of the movies.

# In[23]:


moviemat = df.pivot_table(index='user_id',columns='title', values='rating')


# In[24]:


moviemat.head()


# In[25]:


ratings.sort_values('num of ratings', ascending=False).head(15)


# For my movie recommender I am going to choose two films I enjoyed and that have a large number of ratings: **Fargo** and **Pulp Fiction**

# In[26]:


fargo_user_ratings = moviemat['Fargo (1996)']
pulpfiction_user_ratings = moviemat['Pulp Fiction (1994)']


# In[27]:


fargo_user_ratings.head()


# I use corrwith() method to get correlations between two pandas series:

# In[28]:


similar_to_fargo = moviemat.corrwith(fargo_user_ratings)


# In[29]:


similar_to_pulpfiction = moviemat.corrwith(pulpfiction_user_ratings)


# In[30]:


corr_fargo = pd.DataFrame(similar_to_fargo,columns=['Correlation'])
corr_fargo.dropna(inplace=True)


# In[31]:


corr_fargo.head()


# This correlation column shows how correlated each film's user rating is to the user ratings of **Fargo**. Sorting by correlation:

# In[32]:


corr_fargo.sort_values('Correlation', ascending=False).head(10)


# These films correlate perfectly with **Fargo** but these ratings are most likely from users who've only seen **Fargo** and one or two other films. A larger number of ratings is needed to get a more accurate correlation. This is done by filtering out films that have less than a certain number of reviews.

# In[33]:


corr_fargo = corr_fargo.join(ratings['num of ratings'])


# In[34]:


corr_fargo.head()


# From the histogram earlier, there is a decline in the number of ratings after 100. I need to strike a balance between good correllation but also I need to pick films that enough people have seen.

# In[59]:


corr_fargo[corr_fargo['num of ratings']>75].sort_values('Correlation',ascending=False).head()


# From reading their IMdB summaries, these suggestions seem like good picks as they seem to be the kind of films the Coen Brothers' fans would enjoy. After some tinkering with the filter, I settled on 75.

# Doing the same for **Pulp Fiction**:

# In[36]:


corr_pulpfiction = pd.DataFrame(similar_to_pulpfiction,columns=['Correlation'])


# In[37]:


corr_pulpfiction = pd.DataFrame(similar_to_pulpfiction,columns=['Correlation'])


# In[38]:


corr_pulpfiction.dropna(inplace=True)


# In[39]:


corr_pulpfiction = corr_pulpfiction.join(ratings['num of ratings'])


# In[58]:


corr_pulpfiction[corr_pulpfiction['num of ratings']>75].sort_values('Correlation', ascending=False).head()


# **True Romance** was written by Quentin Tarantino so it seems like a good suggestion. I have seen the other films before and I have enjoyed them.
