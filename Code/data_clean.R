setwd("~/Y2S1/CS3244/3244dataset")

library(tidyverse)
library(readxl)

# List of all stopwords in the language (common words in the English language)
# stopwords("english")

train_df <- read.csv("train.csv/train.csv", stringsAsFactors=FALSE)
train_df <- as_tibble(train_df)

test_df <- read.csv("test.csv/test.csv", stringsAsFactors = FALSE)
test_df <- as_tibble(test_df)

cleanString <- function(x) {
  x <- x %>%
    str_to_lower() %>%
    str_replace_all(pattern="'", replacement=" ") %>%
    str_replace_all(pattern="\n", replacement=" ") %>%
    str_replace_all(pattern="[^a-z|'|[:space:]]", replacement=" ") %>%
    str_replace_all(pattern="[|]", replacement=" ") %>%
    str_replace_all(pattern="\\s+", replacement=" ") %>%
    str_trim()
}

train_df %>%
  mutate(comment_text = sapply(comment_text, checkSpam)) %>%
  write_csv(path="train.csv/train_noSpam.csv")

test_df %>%
  mutate(comment_text = sapply(comment_text, checkSpam)) %>%
  write_csv(path="test.csv/test_noSpam.csv")

checkSpam <- function(x) {
  x <- x %>% 
    str_split(pattern=" ") %>%
    unlist()
  
  strLength <- length(x)
  uniqueWords <- x %>%
    unique() %>%
    length()
  
  # Setting spam threshold to be 0.3
  if (uniqueWords / strLength < 0.3) {
    return(TRUE)
  } else {
    return(FALSE)
  }
}


train_df2 <- train_df %>% 
  mutate(spam = sapply(comment_text, checkSpam)) %>%
  filter(spam == TRUE, toxic == 0)

train_df3 <- train_df %>% 
  mutate(spam = sapply(comment_text, checkSpam)) %>%
  filter(spam == TRUE, toxic == 1)


spam_comments_train <- train_df2 %>%
  filter(spam == TRUE) %>%
  select(comment_text) 

spam_comments_train %>%
  write_csv(path="train.csv/train_spam_comments.csv")

train_df2 %>%
  write_csv(path="train.csv/train_noSpam.csv")

test_df2 <- test_df %>% 
  mutate(spam = sapply(comment_text, checkSpam))

spam_comments_test <- test_df2 %>%
  filter(spam == TRUE) %>%
  select(comment_text)

spam_comments_test %>%
  write_csv("test.csv/test_spam_comemnts.csv")

test_df2 %>%
  write_csv(path="test.csv/test_noSpam.csv")


# Test for spam comments
test_str <- "This is a comment. This is a comment. This is not a comment. Oh wow."
checkSpam(test_str)
