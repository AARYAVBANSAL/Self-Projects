# Projects

## 1. `sudoku/sudoku_solver.cpp`
Recursive backtracking Sudoku solver in C++.
```
g++ -std=c++17 -O2 -o sudoku_solver sudoku_solver.cpp
./sudoku_solver              # built-in sample puzzle
./sudoku_solver puzzle.txt   # your own puzzle (9 lines, '.' or 0 = blank)
```

## 2. `gpt2/gpt2_small.py`
GPT-2 small built from scratch in PyTorch: custom byte-pair encoding
tokenizer, causal self-attention, transformer blocks, training loop, and
autoregressive generation. Math notes in `gpt2/NOTES.md`.
```
pip install torch --break-system-packages
python gpt2_small.py --train sample.txt --steps 2000 --generate "Once upon a time"
```

## 3. `pairtrading/pair_trading_strategy.py`
Market-neutral pairs trading strategy: polynomial-regression spread model,
ADF + Engle-Granger cointegration tests, backtested with Backtrader/Cerebro.
```
pip install backtrader statsmodels yfinance numpy pandas --break-system-packages
python pair_trading_strategy.py --tickers KO PEP --start 2018-01-01 --end 2024-01-01
```
