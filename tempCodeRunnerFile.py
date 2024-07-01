df['Close'].expanding(min_periods=1).mean().plot(figsize=(13, 3))
