# Movie-genre classifier
Multi-modal analysis of plot and poster of movie to classify genre

## Flow of Project

1. Data scraping
2. Data preprocessing / cleaning
3. Text Analysis using LDA model (guided LDA model)
4. Poster Analysis using CNN (Convolutional Neural Network)
5. Combining two models
6. Result

## Detail of Model

We were able to scrape ___ number of movies' plot and poster.

There were in total ___ number of genres existing in our database.

There were at least _____ number of movies in each genre.

1. Data scraping

> Scraping imdb.com using Beautifulsoup

2. Data preprocessing / cleaning

> Text preprocessing (Removal of stopwords, lemmatization)

> Label image to genre
  
3. Text Analysis using LDA model (guided LDA model / LDA)

> LDA analysis without seeding

> Seeds of model / How did we chose them? 
  
4. Poster Analysis using CNN (Convolutional Neural Network)
  
>   
>   
  
5. Combining two models

>  Ensembled learning ? 

>  
  
## Result of Model

We have splitted our dataset into test data and train data. 

When feeding our test data into our model we have attained ____ % accuracy. 

By looking at our confusion matrix, we notice that the model _____.


https://datascience.stackexchange.com/questions/1123/combine-multiple-classifiers-to-build-a-multi-modal-classifier
https://medium.com/appliedalgoritmo/an-intro-to-ensemble-learning-in-machine-learning-5ed8792af72d
https://medium.com/diogo-menezes-borges/ensemble-learning-when-everybody-takes-a-guess-i-guess-ec35f6cb4600
https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164
