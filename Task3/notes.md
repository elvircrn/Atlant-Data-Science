
Links:
[Model-based CF](http://www.cs.carleton.edu/cs_comps/0607/recommend/recommender/modelbased.html)
[One-Hot vs integer](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/)
[Matrix Factorization](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/)
[Wikia](http://stat.wikia.com/wiki/Matrix_factorization)
[C++ SVD RS](https://github.com/timnugent/svd-recommend)
[Another C++ RS](https://github.com/yixuan/recosystem)
[SVD++](http://www.recsyswiki.com/wiki/SVD%2B%2B)
[Basic Clustering Model](http://cs229.stanford.edu/proj2013/Bystrom-MovieRecommendationsFromUserRatings.pdf)
[Blog](http://sifter.org/~simon/journal/20061211.html)
[SVD & PMF Ppt](http://dparra.sitios.ing.uc.cl/classes/recsys-2015-2/student_ppts/CRojas_SVDpp-PMF.pdf)
[SVD vs SVD++](https://www.quora.com/Whats-the-difference-between-SVD-and-SVD++)
[Netflix clustering](https://rpubs.com/nishantsbi/93582)
[ALS](https://bugra.github.io/work/notes/2014-04-19/alternating-least-squares-method-for-collaborative-filtering/)
# Dataset Preprocessing and Analysis 

Max movie id: 176279
Max user id: 270896

normalize:
movieId
gtags
director name

## Steps:

### movie meta
1. fillna
2. map str to ind
3. normalize
4. movie str preprocess
5. binary encode genres
6. map plot keywords
7. delete useless

## movies.csv
movieId

title

genres

## genome-tags.csv
tagId

tag

## ratings.csv
userId

movieId

rating
timestamp

## tags.csv
userId

movieId

tag

timestamp

## genome-scores.csv
movieId

tagId

relevance

1.
2.
3.

## links.csv
movieId

imdbId

tmdbId

## Movie Metadata

### movie\ metadata.csv
color
map to index
nan -> 0

### director name
map to index
na -> 0

### num critic for reviews
na -> 0

### duration
given in minutes
na -> avg

### director facebook likes
na -> 0

### actor 3 facebook likes
na -> 0

### actor 2 name
na -> 0

### actor 1 facebook likes
na -> 0

### gross
na -> 0

### genres
binary encode

### actor 1 name
map to index

### movie title
map to index
delete?

### num voted users
nan -> 0

### cast total facebook likes
nan -> 0

### actor 3 name
map to index
na -> 0

### facenumber in poster
facenumber??
nan -> 0

### plot keywords
``` py
    keywords = set()

    for plot_keywords in movie_meta['plot_keywords'].astype(str).apply(lambda x: x.split('|')):
        for keyword in plot_keywords:
            keywords.add(keyword)
```
8087 plot keywords in dataset.
Max 5 keywords per movie.
fillna to 'none'
astype(str)

### movie imdb link
Extract id.

### num user for reviews
nan -> 0

### language
map str to ind

### country
map str to ind

### content rating
map str to ind


### budget
nan -> 0
normalize:
    * number of digits(log10)


### title year
Min: 1916
year - 1915
nan -> 0


### actor 2 facebook likes
nan -> 0


### imdb score
Normalize to [0.0, 1.0]


### aspect ratio
delete

### movie facebook likes
na -> 0


