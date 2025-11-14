# Method Comparison: Metrics Summary

| Method              |   Weekly MAE |   Pearson Corr |   NMAE | Bias %   |   Alpha CV |
|:--------------------|-------------:|---------------:|-------:|:---------|-----------:|
| Baseline            |         0.73 |          0.906 |  0.803 | 75.0%    |       1.13 |
| Hierarchical        |         0.31 |          0.954 |  0.337 | 11.4%    |       0.74 |
| Smooth Alpha (λ=10) |         0.31 |          0.954 |  0.336 | 11.4%    |       0.74 |
| Hierarchical+DOW    |         0.33 |          0.953 |  0.362 | 17.8%    |       0.74 |

**Note**: Weekly MAE < 1.5 is the target. Smooth Alpha achieves 0.32, exceeding the target by 78%.
