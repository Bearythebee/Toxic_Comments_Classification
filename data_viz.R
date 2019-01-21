setwd("~/Y2S1/CS3244/3244dataset")

library(tidyverse)
library(readxl)
library(tm)
library(wordcloud)

# clean_train_df <- clean_train_df[sample(nrow(clean_train_df), 10000), ]

train_df <- read_csv("train.csv/train.csv") %>%
  filter(insult != 0) 
# train_df <- train_df[sample(nrow(train_df), 1000), ]

# |severe_toxic != 0|obscene != 0|threat != 0|insult != 0|
#   identity_hate != 0

corpus <- Corpus(VectorSource(train_df$comment_text))
corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeWords, stopwords("english")) %>%
  tm_map(stripWhitespace) %>%
  tm_map(removeWords, c("wikipedia", "article", "page", "thanks", "also", "really",
                        "just", "will", "please", "pages", "can", "talk", "like",
                        "need", "thank"))

tdm <- TermDocumentMatrix(corpus)
m <- as.matrix(tdm)
v <- sort(rowSums(m), decreasing=TRUE)
d <- data.frame(word=names(v), freq=v)

# wordcloud(d$word, d$freq, random.order=FALSE, max.words=100, 
#           colors=brewer.pal(8, "Dark2"), scale=c(2.5,0.5))
# 
# ?wordcloud

d2 <- d[1:15, ]
d2

d2 <- d2 %>%
  mutate(word = reorder(word, freq)) 
ggplot(d2) +
  geom_bar(aes(y = freq, x= word), stat="identity", fill="indianred") +
  labs(x="Word", y="Number of appearances", title='Top 15 words in "insult" class') +
  coord_flip()
