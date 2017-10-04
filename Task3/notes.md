# Dataset Preprocessing and Analysis


## movies.csv
movieId

title

genres

## genome-tags.csv
tagId

tag

## genome-scores.csv
movieId

tagId

relevance

## links.csv
movieId

imdbId

tmdbId

## Movie Metadata

### ratings.csv
userId,movieId,rating,timestamp

### tags.csv
userId,movieId,tag,timestamp

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


