#!/usr/bin/env python
# coding: utf-8

# # Analyze A/B Test Results 
# 
# 
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# Now, we read our 'ab_data' file

# In[2]:


df= pd.read_csv('ab_data.csv')
df.head()


# **b.** Use the cell below to find the number of rows in the dataset.

# In[3]:


df.info()


# So, we have 294478 rows

# In[4]:


df.shape


# We have 294478 rows and 5 columns

# **c.** The number of unique users in the dataset.

# In[5]:


len(pd.unique(df['user_id']))


# We have 290584 unique users

# **d.** The proportion of users converted.

# In[6]:


df.query('converted == 1').count()[0]/ df.shape[0]


# **e.** The number of times when the "group" is `treatment` but "landing_page" is not a `new_page`.

# This is obtained as follows:

# In[7]:


a =df[(df['group'] == 'treatment') & (df['landing_page'] != 'new_page')].count()[0]
b = df[(df['group'] == 'control') & (df['landing_page'] != 'old_page')].count()[0]
a + b


# **f.** Do any of the rows have missing values?

# In[8]:


df.isnull().count()


# There is no row having missing values

# 
# **a.** Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# We will remove the inaccurate rows, and store the result in a new dataframe df2

# In[9]:


d1 =df[(df['group'] == 'treatment') & (df['landing_page'] == 'new_page')]
d2 =df[(df['group'] == 'control') & (df['landing_page'] == 'old_page')]
s = [d1,d2]
d2.shape

        


# Now, we concat s

# In[10]:


df2= pd.concat(s)
df2.head()


# We will attempt to double check, if all of the incorrect rows were removed from df2

# In[11]:


df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# **a.** How many unique **user_id**s are in **df2**?

# In[12]:


len(pd.unique(df2['user_id']))


# There are 290584 unique_ids in df2

# **b.** There is one **user_id** repeated in **df2**.  What is it?

# In[13]:


df2['user_id'].mode()


# That is definitely user - 773192

# **c.** Display the rows for the duplicate **user_id**? 

# In[14]:


df2.loc[df2['user_id']== 773192]


# Those are the duplicate rows

# **d.** Remove **one** of the rows with a duplicate **user_id**, from the **df2** dataframe.

# In[15]:


df2= df2.drop_duplicates(subset = 'user_id', keep = 'first')


# To re-check again if the row with a duplicate user_id is deleted or not
# 

# In[16]:


df2.loc[df2['user_id']== 773192]


# Great!!! It has been deleted

# Now, let us check the shape of our dataset

# In[17]:


df2.shape


# .
# 
# **a.** What is the probability of an individual converting regardless of the page they receive?<br><br>
# 
# 
# 
# 

# In[18]:


df2.head(3)


# In[19]:


df2['converted'].sum()/290584


# The probability an individual converts regardless of the page they recieve is 0.1196

# **b.** Given that an individual was in the `control` group, what is the probability they converted?

# In[20]:


df2.groupby(['group', 'converted']).count()


# In[21]:


Pc=17489/145274
Pc


# Given an indidual is in the control group, the probability they converted is 0.1204

# **c.** Given that an individual was in the `treatment` group, what is the probability they converted?

# In[22]:


Pt=17264/145310
Pt


# Given an indidual is in the treatment group, the probability they converted is 0.1188

# Now, we calculate the actual difference (obs_diff) between the conversion rates for the two groups.

# In[23]:


obs_diff= df2[df2['group']=='treatment']['converted'].mean() - df2[df2['group']=='control']['converted'].mean()
obs_diff


# So, our obs_diff is -0.00158

# **d.** What is the probability that an individual received the new page?

# In[24]:


df2.groupby(['landing_page']).count()


# From the above groupby method, the probability an individual recieved a new page is 0.5

# In[25]:


145310/290584


# **e.** Consider your results from parts (a) through (d) above, and explain below whether the new `treatment` group users lead to more conversions.

# Looking at the convertion rate (Pc and Pt), it can be easily said that the new treatment group does not lead to more convertion rate as Pc>Pt. However, we also have to take into consideration the number of individuals in each group before making our conclusion. 
# From the analysis above, we saw that the number of indivuals in the control group is 145274 and that of the treatment group is 145310. 
# As the different is not that much, we can infer that the above conclusion is valid: the control group has more convertion rate. 

# If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should be your null and alternative hypotheses (**$H_0$** and **$H_1$**)?  
# 
# You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the "converted" probability (or rate) for the old and new pages respectively.

# Anwser
#                            𝑝new <= pold
# 
#                            𝑝new > pold
#                            
#                            or
#                            
#                            𝑝new - pold <= 0
#                            
#                            𝑝new - pold > 0

# ### ToDo 2.2 - Null Hypothesis $H_0$ Testing
# 

# **a.** What is the **conversion rate** for $p_{new}$ under the null hypothesis? 

# In[26]:


Pnew = Pold =df2['converted'].sum()/290584
Pnew


# The conversion rate for Pnew is 0.1196

# **b.** What is the **conversion rate** for $p_{old}$ under the null hypothesis? 

# In[27]:


Pnew = Pold =df2['converted'].sum()/290584
Pold


# The conversion rate for Pold is also 0.1196

# **c.** What is $n_{new}$, the number of individuals in the treatment group? <br><br>
# 

# In[28]:


nnew = df2[df2['group']=='treatment'].count()[0]
nnew


# The number of individuals in the treatment group is 145310

# **d.** What is $n_{old}$, the number of individuals in the control group?

# In[29]:


nold = df2[df2['group']=='control'].count()[0]
nold


# The number of individuals in the control group is 145274

# **e. Simulate Sample for the `treatment` Group**<br> 
# Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null hypothesis.  <br><br>
# 

# We simulate a Sample for the treatment Group

# In[35]:


new_page_converted = np.random.choice([0,1], size = nnew)
new_page_converted


# **f. Simulate Sample for the `control` Group** <br>
# Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null hypothesis. <br> 

# We simulate a Sample for the control Group

# In[36]:


old_page_converted = np.random.choice([0,1], size = nold)
old_page_converted


# **g.** Find the difference in the "converted" probability $(p{'}_{new}$ - $p{'}_{old})$ for your simulated samples from the parts (e) and (f) above. 

# Let "pUnew" be the treatment group converted probability of the simulated sample above

# In[37]:


pUnew = new_page_converted.mean()
pUnew


# Let "pUold" be the treatment group converted probability of the simulated sample above

# In[38]:


pUold = old_page_converted.mean()
pUold


# Let dU be the difference in the two converted probabilities, then: 

# In[39]:


dU = pUnew- pUold
dU


# 
# **h. Sampling distribution** <br>
# Re-create `new_page_converted` and `old_page_converted` and find the $(p{'}_{new}$ - $p{'}_{old})$ value 10,000 times using the same simulation process you used in parts (a) through (g) above. 
# 
# <br>
# Store all  $(p{'}_{new}$ - $p{'}_{old})$  values in a NumPy array called `p_diffs`.

# We create the sampling distribution as follows:

# In[40]:


p_diffs = []
for _ in range (10000):
    new_page_converted = np.random.choice([0,1], size = nnew, replace = True, p = [Pnew, 1-Pnew] )
    old_page_converted = np.random.choice([0,1], size = nold, replace = True, p = [Pold, 1-Pold])
    pUnew = new_page_converted.mean()
    pUold = old_page_converted.mean()
    dU = pUnew- pUold
    p_diffs.append(dU)
    


# **i. Histogram**<br> 
# Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.<br><br>
# 

# We create our histogram as follows:

# In[41]:


plt.hist(p_diffs)
plt.axvline(x=obs_diff, color = 'red')
plt.ylabel('Frequency')
plt.xlabel('Pnew-Pold')


# As seen above, our histogram looks like what we expected. 

# **j.** What proportion of the **p_diffs** are greater than the actual difference observed in the `df2` data?

# The required proportion is obtained as follows:

# In[43]:


(p_diffs > obs_diff).mean()


# **k.** Please explain in words what you have just computed in part **j** above.  
#  - What is this value called in scientific studies?  
#  - What does this value signify in terms of whether or not there is a difference between the new and old pages? 

# Anwsers:
# 1. The P_value
# 2. Since our P_value(0.9029) is greater than our Type 1 error rate(0.05), we will fail to reject the null hypothesis. This implies that pnew <= Pold

# 
# 
# **l. Using Built-in Methods for Hypothesis Testing**<br>
# We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. 
# 

# In[44]:


import statsmodels.api as sm

# number of conversions with the old_page
convert_old = df2[df2.group == 'control'].converted.sum()

# number of conversions with the new_page
convert_new =df2[df2.group == 'treatment'].converted.sum()

# number of individuals who were shown the old_page
n_old = df2[df2['landing_page'] == 'old_page'].count()[0]

# number of individuals who received new_page
n_new = df2[df2['landing_page'] == 'new_page'].count()[0]


# In[45]:


convert_old, convert_new, n_old, n_new


# **m.** Now use `sm.stats.proportions_ztest()` to compute your test statistic and p-value
# 
# 

# In[46]:


count_array = np.array([convert_new, convert_old])
nobs_array = np.array([n_new,n_old])


# In[47]:


import statsmodels.api as sm
z_score, p_value = sm.stats.proportions_ztest(count_array, nobs_array, alternative = 'larger'  )
print(z_score, p_value)


# Therefore, our z_score is -1.311 and our P_value is 0.9051

# **n.** What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?<br><br>
# 
# 

# The p_value obtained here is similar to that obtained in section (2.2 j) above. Since the p_values are both greater than the Type 1 error rate, we will fail to reject the null hypothesis. So, pnew <= Pold

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# 
# 
# **a.** Since each row in the `df2` data is either a conversion or no conversion, what type of regression should you be performing in this case?

# Answer: Multiple Linear Regression

# **b.** The goal is to use **statsmodels** library to fit the regression model you specified in part **a.** above to see if there is a significant difference in conversion based on the page-type a customer receives. However, you first need to create the following two columns in the `df2` dataframe:
#  1. `intercept` - It should be `1` in the entire column. 
#  2. `ab_page` - It's a dummy variable column, having a value `1` when an individual receives the **treatment**, otherwise `0`.  

# In[48]:


df2.head(2)


# The first thing to do is to create our intercept as follows:

# In[49]:


df2['intercept'] = 1


# Then, we create our dummy variable for the 'landing_page' colummn, using 'old_page' as baseline as follows:

# In[50]:


df2[['new_page', 'old_page']] = pd.get_dummies(df['landing_page'])


# Done!!!

# In[51]:


df2.head(2)


# **c.** Use **statsmodels** to instantiate your regression model on the two columns you created in part (b). above, then fit the model to predict whether or not an individual converts. 
# 

# We fit in our regression model like this:

# In[52]:


import statsmodels.api as sm
lm = sm.OLS(df2['converted'], df2[['intercept','new_page']])
results= lm.fit()


# **d.** Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[53]:


results.summary()


# **e.** What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  
# 
# **Hints**: 
# - What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**? 
# - You may comment on if these hypothesis (Part II vs. Part III) are one-sided or two-sided. 
# - You may also compare the current p-value with the Type I error rate (0.05).
# 

# Answer
# The p_value associated with ab_page(new_page) is 0.190.
# If differs from that obtained in Part 11 because:Part II uses z-test statistics to calculate the p-value and Part III uses logistic regression which calculates the p-value using t-test statistics.
# 
# Te difference between the z-test and the t-test (and consequent p-value calculation) is that the z-test was conducted with one-tailed test, while the t-test for logistic regression was a two tailed test.
# 
# The p_value of 0.190 is also greater than the Type 1 error rate(0.05), therefore, this suggests that the variable 'new_page' is not statistially significant in relating to the response variable(i.e convertibility)

# **f.** Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# Answer:
# In a real life situation, the factors that determines whether or not an indiviual converts are much more than those considered in the analysis above. Thus, a given analysis might not be totally accurate(biased) if those underlying factors are not considered.
# 
# However, if too many factors are being considered for a given analysis, it could lead to a problem of multicollinearity.

# **g. Adding countries**<br> 
# Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. 
# 
# 
# 
#  Provide the statistical output as well as a written response to answer this question.

# Let us read our 'countries' data

# In[54]:


dfc = pd.read_csv('countries.csv')
dfc.head(5)


# Now, we will attempt to merge this 'countries' dataset with our dataset df2 in df3 like this:

# In[55]:


df3= pd.merge(df2, dfc, on ='user_id')
df3.head()


# Now we will create dummy variables for the country column, using 'CA' as baseline

# In[56]:


df3[['US', 'UK', 'CA']] = pd.get_dummies(df3['country'])


# **h. Fit your model and obtain the results**<br> 
# Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if are there significant effects on conversion. 
# 

# First of all, let us create an interaction between the page and country variable as follows:

# In[57]:


df3['new_page_UK'] = df3['new_page'] * df3['UK']
df3['new_page_US'] = df3['new_page'] * df3['US']


# We fit your model, and summarize the results

# In[58]:



df3['intercept'] = 1
lm = sm.Logit(df3['converted'], df3[['intercept', 'new_page', 'UK' , 'US', 'new_page_UK', 'new_page_US']])
results= lm.fit()
results.summary2()


# Anwser:
# Considering the P_values with a 0.05 Type 1 error rate:
# As the P-values for new_page(0.1323), UK(0.7598), US(0.6418), new_page_UK(0.2377) and new_page_US(0.3833) are all greater than 0.05, it means that none of this variables is statistically significant for predicting the conversion rate of individuals. We will fail to reject the null hypothesis. 
# 
# In practical sense, it means that neither the location of an individual, nor the individual's landing page can be used to predict whether or not the individual will convert or not.  
# 
# 
# 

# <a id='finalcheck'></a>
# ## Final Check!
# 
# Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your notebook to make sure that it satisfies all the specifications mentioned in the rubric. You should also probably remove all of the "Hints" and "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# <a id='submission'></a>
# ## Submission
# You may either submit your notebook through the "SUBMIT PROJECT" button at the bottom of this workspace, or you may work from your local machine and submit on  the last page of this project lesson.  
# 
# 1. Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# 
# 2. Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# 
# 3. Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[60]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




