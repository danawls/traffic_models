setwd("/Users/danawls/Desktop/*Important*/traffic-deep-learning-research/table-figure/figure/traffic_cost")
origin_data <- read.csv("yearly_data.csv")

library(ggplot2)

p <- ggplot(origin_data, aes(x = Year, y = Traffic_cost)) +
       geom_line(size = 0.5) +
       labs(title = "yearly traffic cost", x = "year", y = "traffic_cost")

p

ggsave("graph.png", plot = p)
