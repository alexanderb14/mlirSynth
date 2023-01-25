library(ggplot2)
library(ggdark)
library(scales)
library(dplyr)

# Prevent Rplots.pdf from being created
pdf(NULL)

args = commandArgs(trailingOnly=TRUE)
if (length(args) > 0) {
  inputfile <- args[1]
  } else {
    inputfile <- "2/stats.csv"
}
data <- read.csv(inputfile)

out_dir <- "/tmp/plots/"
if (!file.exists(out_dir)) {
  dir.create(out_dir)
}

breaks = c(0, 1, 10, 100)
breaks_detailed = unique(c(seq(0, 1, 0.1),
                           seq(1, 10, 1),
                           seq(10, 100, 10),
                           seq(100, 300, 100)))

# General
ggplot(data, aes(x = benchmark, y = synth_time, fill = operations)) +
  geom_col(position = "dodge") +
  facet_grid("distribute ~ prune_equivalent_candidates", labeller = "label_both") +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
    breaks = breaks,
    labels = breaks
  ) +
  geom_hline(yintercept = 1) +
  geom_hline(yintercept = 300) +
  dark_theme_gray()
ggsave(paste(out_dir, "overview.pdf", sep = ""))

# Feature: prune_equivalent_candidates
ggplot(data,
       aes(x = benchmark, y = synth_time, fill = prune_equivalent_candidates)) +
  geom_col(position = "dodge") +
  facet_wrap("distribute", labeller = "label_both") +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
    breaks = breaks_detailed,
    labels = breaks_detailed
  ) +
  geom_hline(yintercept = 1) +
  geom_hline(yintercept = 300) +
  dark_theme_gray()
ggsave(paste(out_dir, "feature_distribute.pdf", sep = ""))

# Feature: distribute
ggplot(data, aes(x = benchmark, y = synth_time, fill = distribute)) +
  geom_col(position = "dodge") +
  facet_wrap("prune_equivalent_candidates", labeller = "label_both") +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
    breaks = breaks_detailed,
    labels = breaks_detailed
  ) +
  geom_hline(yintercept = 1) +
  geom_hline(yintercept = 300) +
  dark_theme_gray()
ggsave(paste(out_dir, "feature_prune_equivalent_candidates.pdf", sep = ""))

# All best
data_best <-
  data %>% group_by(benchmark, operations) %>% summarise(synth_time_best = min(synth_time))

ggplot(data_best,
       aes(x = benchmark, y = synth_time_best, fill = operations)) +
  geom_col(position = "dodge") +
  scale_y_continuous(
    trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
    breaks = breaks_detailed,
    labels = breaks_detailed
  ) +
  geom_hline(yintercept = 1) +
  geom_hline(yintercept = 300) +
  dark_theme_gray()
ggsave(paste(out_dir, "all_best.pdf", sep = ""))

