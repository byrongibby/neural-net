library(idx2r)
library(R.utils)

url <- "http://yann.lecun.com/exdb/mnist/"



# Training images ---


file_name <- "train-images-idx3-ubyte.gz"

download.file(paste0(url, file_name), file_name)
gunzip(file_name)

train_images <- read_idx(gsub(pattern = "\\.gz", "", file_name))

unlink(gsub(pattern = "\\.gz", "", file_name))



# Training labels ---


file_name <- "train-labels-idx1-ubyte.gz"

download.file(paste0(url, file_name), file_name)
gunzip(file_name)

train_labels <- read_idx(gsub(pattern = "\\.gz", "", file_name))

unlink(gsub(pattern = "\\.gz", "", file_name))



# Training images ---


file_name <- "t10k-images-idx3-ubyte.gz"

download.file(paste0(url, file_name), file_name)
gunzip(file_name)

test_images <- read_idx(gsub(pattern = "\\.gz", "", file_name))

unlink(gsub(pattern = "\\.gz", "", file_name))



# Test labels ---


file_name <- "t10k-labels-idx1-ubyte.gz"

download.file(paste0(url, file_name), file_name)
gunzip(file_name)

test_labels <- read_idx(gsub(pattern = "\\.gz", "", file_name))

unlink(gsub(pattern = "\\.gz", "", file_name))



# Save data ---


save(train_images, train_labels, file = file.path("data", "train.RData"))
save(test_images, test_labels, file = file.path("data", "test.RData"))
