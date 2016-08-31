# Easy Latent Dirichlet Allocation

The ultimate goal of this project is to take something like a bunch of poorly formatted, grammatically challenged 
internet movie reviews, and output a list of usable topics, like `[u'film movie wa action alien star first effect 
war', u'american life war men power black political art state', u'comedy funny john high big little laugh fun star', u'life story first family little world character ha one', u'film one movie ha wa like character time scene']`

Frustrated by the complexity and time sink of using common Python tools to perform topic modeling, I've iterated on 
this code over the last few years as a tool to quickly:

 - clean text (if necessary)
 - train an LDA model
 - predict a topic for text

I've mimicked the SKLearn fit(), transform() interface, operating on numpy objects, to make this easier to use. I've 
also included `predict_proba`, `get_topic_descriptions`, `preprocess`, and `transform_with_all_data_df` to make 
utilizing your model easier.  

## Getting started

### Repo structure
The actual lda class is in `bin/easy_lda`, and an example of using it is in `bin/lda_example`. This code does have 
some pre-requisite modules, found in `environment.yml` (see _Python Environment_ for more). 

### Python Environment
Python code in this repo utilizes packages that are not part of the common library. To make sure you have all of the 
appropriate packages, please install [Anaconda](https://www.continuum.io/downloads), and install the environment 
described in environment.yml (Instructions [here](http://conda.pydata.org/docs/using/envs.html), under *Use 
environment from file*, and *Change environments (activate/deactivate)*). 

### To run code
  
To run the Python code, complete the following:
```bash
# Install anaconda environment
conda env create -f environment.yml 

# Make a note of the environent name (the console will show something like: source activate _environment_name_)

# Activate environment
source activate environment_name

# Run script
cd bin/
python lda_example.py
```

### Optimizing models

Not happy with the default settings? Here are a few areas to tweak to get more meaningful topics:

 - Number of topics (`num_topics`): Depending on the number of documents, and how discrete your topics are, you can try
  increasing 
 or decreasing the number of topics. My rule of thumb is usually 3-10 topics, with a strong preference for less topics. 
 
 - Number of iterations (`num_iterations`): Some datasets take more time to converge. There is little harm in 
 increasing the number of iterations (other than waiting longer), but there is some damage in halting too early. My 
 rule of thumb is to wait for the log likelihood to vary by less than ~.1% before saying the topics are stable. 
 
 - Stop words (not currently exposed): Many words are common, but not meaningful (examples include ['the', 'and', 
 'where']). The built in data cleaning method (`preprocess`) takes care of most common English language stop words, 
 but there may be some that are unique to your data set (for example, movie is likely a stop word for the data set 
 used in `bin/lda_example.py`). Try implementing your own variaiton of `preprocess`, with more or less stop words. 

## Contact
Feel free to contact me at 13herger <at> gmail <dot> com
