# Method Comparison: Metrics Summary

| Method               |   Weekly MAE |   Pearson Corr |   NMAE | Bias %   |   Alpha CV |
|:---------------------|-------------:|---------------:|-------:|:---------|-----------:|
| Baseline             |         0.7  |          0.936 |  0.765 | 74.4%    |       0.72 |
| Hierarchical         |         0.36 |          0.902 |  0.391 | 3.6%     |       0.81 |
| Smooth Alpha (λ=499) |         0.26 |          0.937 |  0.279 | -0.4%    |       0.54 |
| Hierarchical+DOW     |         0.38 |          0.9   |  0.412 | 9.2%     |       0.81 |

**Note**: Weekly MAE < 1.5 is the target. Smooth Alpha achieves 0.32, exceeding the target by 78%.
