#!/usr/bin/env Rscript

# File paths defined at the top for easy modification
df_wide_path <- "data.csv"
time_series_path <- "time.csv"

# Load required package
library(grofit)

# Validate input files
for (file_path in c(df_wide_path, time_series_path)) {
  if (!file.exists(file_path) || file.info(file_path)$size == 0) {
    stop(paste("Error: File", file_path, "not found or empty"))
  }
}

# Read data files
df_wide <- read.csv(df_wide_path, header = FALSE, stringsAsFactors = FALSE)
time_series <- read.csv(time_series_path, header = FALSE, stringsAsFactors = FALSE)

# Validate data content
if (nrow(df_wide) == 0 || ncol(df_wide) == 0) {
  stop(paste("Error:", df_wide_path, "has no data or columns"))
}

# Process data and run grofit
time_series <- as.matrix(time_series)
fit_result <- gcFit(time_series, df_wide, control = grofit.control(interactive=FALSE))

# Save summary to file
sum <- summary.gcFit(fit_result)
write.csv(sum, "summary.csv", row.names = FALSE)

