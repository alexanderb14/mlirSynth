library(ggplot2)
library(ggdark)
library(scales)
library(dplyr)

# Prevent Rplots.pdf from being created
pdf(NULL)

args = commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  inputfile <- args[1]
} else {
  inputfile <- "2/stats.csv"
}
data <- read.csv(inputfile)

if (length(args) > 1) {
  PLOT_DARK = FALSE
} else {
  PLOT_DARK = TRUE
}

out_dir <- "/tmp/plots/"
if (!file.exists(out_dir)) {
  dir.create(out_dir)
}

breaks = c(0, 1, 10, 100)
breaks_detailed = unique(c(seq(0, 1, 0.1),
                           seq(1, 10, 1),
                           seq(10, 100, 10),
                           seq(100, 500, 100)))

plot_and_save <- function(plot_obj, filename) {
  if (PLOT_DARK == TRUE) {
    plot_obj = plot_obj + dark_theme_gray()
  }
  plot_obj = plot_obj + theme(axis.text.x = element_text(
    angle = 45,
    vjust = 0.5,
    hjust = 0.5
  ))

  print(plot_obj)
  ggsave(paste(out_dir, filename, sep = ""),
         width = 12,
         height = 8)
}
dev.off()

# General
plot_and_save(
  ggplot(data, aes(
    x = benchmark, y = synth_time, fill = operations
  )) +
    geom_col(position = "dodge") +
    facet_grid("distribute ~ prune_equivalent_candidates", labeller = "label_both") +
    scale_y_continuous(
      trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
      breaks = breaks,
      labels = breaks
    ) +
    geom_hline(yintercept = 1) +
    geom_hline(yintercept = 500),
  "overview.png"
)

# Feature: prune_equivalent_candidates
plot_and_save(
  ggplot(
    data,
    aes(x = benchmark, y = synth_time, fill = prune_equivalent_candidates)
  ) +
    geom_col(position = "dodge") +
    facet_wrap("distribute", labeller = "label_both") +
    scale_y_continuous(
      trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
      breaks = breaks_detailed,
      labels = breaks_detailed
    ) +
    geom_hline(yintercept = 1) +
    geom_hline(yintercept = 500),
  "feature_distribute.png"
)

# Feature: distribute
plot_and_save(
  ggplot(data, aes(
    x = benchmark, y = synth_time, fill = distribute
  )) +
    geom_col(position = "dodge") +
    facet_wrap("prune_equivalent_candidates", labeller = "label_both") +
    scale_y_continuous(
      trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
      breaks = breaks_detailed,
      labels = breaks_detailed
    ) +
    geom_hline(yintercept = 1) +
    geom_hline(yintercept = 500),
  "feature_prune_equivalent_candidates.png"
)

# All best
data_best <-
  data %>% group_by(benchmark, operations)%>% summarise(synth_time_best = min(synth_time, na.rm = TRUE))
data_best %>%
  group_by(operations) %>%
  summarise(ymean = mean(synth_time_best))

plot_and_save(
  ggplot(
    data_best,
    aes(x = benchmark, y = synth_time_best, fill = operations)
  ) +
    geom_col(position = "dodge") +
    scale_y_continuous(
      trans = pseudo_log_trans(sigma = 0.1, base = exp(1)),
      breaks = breaks_detailed,
      labels = breaks_detailed
    ) +
    geom_hline(yintercept = 1) +
    geom_hline(yintercept = 500),
  "all_best.png"
)
