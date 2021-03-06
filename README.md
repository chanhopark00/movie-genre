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

We were able to scrape 45,000 number of movies' plot and poster.

There were in total 20 number of genres existing in our database.


1. Data scraping

> Scraping imdb.com using Beautifulsoup

2. Data preprocessing / cleaning

> Text preprocessing (Removal of stopwords, lemmatization)

> Label image to genre
  
  
3. Poster Analysis using CNN (Convolutional Neural Network)
  
>   
>   
  
4. Combining models
  
 We have splitted our dataset into test data and train data. 

We have made use of multiple models with different measures on the dataset.

### First model, Vanlia version 

<img src="https://github.com/chanhopark00/movie-genre/blob/master/images/1.PNG" width="400" >

### Second model, Class weight added to resolve overfit 

<img src="https://github.com/chanhopark00/movie-genre/blob/master/images/2.PNG" width="400" >

### Third model, SMOTE version 

<img src="https://github.com/chanhopark00/movie-genre/blob/master/images/3.PNG" width="400" >

## Result of Model

Our final model makes use of the average vector generated by each model. 

<img src="https://github.com/chanhopark00/movie-genre/blob/master/images/4.PNG" width="400" >

When feeding our test data into our model we have attained 60 % accuracy. 

<img src="https://github.com/chanhopark00/movie-genre/blob/master/images/5.PNG" width="400" >

By looking this is the heatmap of different genres of movies occuring together; examples such as thriller and action occur together often. Therefore we believe that making use of only one prediction value to compare with the true value is intuitively meaningful.



