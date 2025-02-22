# MADM

Have you ever tried to compare a couple of things and got stuck?

The options seem logical, but you can’t choose the final one?

I know, **it takes Math to choose!**

## What is this

This is a simple implementation of well-known methods for **multi-attribute decision making (MADM)**. You can use it in your day-to-day tasks or for bigger decisions such as buying a house.

## What it can do

### Matrix Preprocessing

Before making a final decision, you need to normalize your decision matrix to avoid having multiple metrics mixed together. This code can help you with that:

- **Normalizes** decision matrix using three different methods:
  - **Linear Normalization**
  - **Euclidean Normalization**
  - **Max-Min Normalization**

### Methods

After normalization, you have multiple ways to find your best option. One of them is the Permutation Method, which is effective but computationally expensive. We strongly advise not using more than 5 attributes with this method—if you have more than 5, consider using one of the other methods.

- Implements the **Permutation Method** to calculate:
  - Concordance matrices
  - Net scores for different permutations of alternatives

Then, there’s the good old **TOPSIS Method**, which ranks alternatives based on their closeness to ideal solutions:

- Implements the **TOPSIS Method** to rank alternatives based on closeness to ideal solutions.

If you prefer a more straightforward approach, you can use **Simple Additive Weighting (SAW)**, which is one of the most intuitive decision-making techniques:

- Implements the **SAW Method**, which scores alternatives by summing their weighted normalized values.

For more rigorous decision-making, you can apply the **ELECTRE I Method**, which is designed for eliminating weaker alternatives:

- Implements the **ELECTRE I Method** to compare alternatives based on concordance and discordance indices.

If you’re a data enthusiast and want your results and calculations saved, we’ve got you covered! All results are outputted into a text file for easy analysis.

- Outputs results into a text file for easy analysis.

### How These Methods Work:

Each of these methods follows a structured approach:

1. **Normalization**: Transform raw values into a comparable scale.
2. **Weighting**: Assign importance levels to different criteria.
3. **Calculation**:
   - **SAW**: Sum the weighted scores for each alternative.
   - **TOPSIS**: Measure distances from the ideal and worst solutions.
   - **Permutation Method**: Compare alternatives using pairwise concordance and net scores.
   - **ELECTRE I**: Use concordance and discordance thresholds to eliminate weaker choices.
4. **Ranking**: Identify the best alternative based on the chosen method.

Each method has its strengths—**SAW** is simple but effective, **TOPSIS** balances ideal and worst-case scenarios, **Permutation Method** provides a thorough comparison, and **ELECTRE I** is great for eliminating poor options.

## What you need beforehand

- Python
- `pandas` library
- `numpy` library

## Disclaimer

This project contains implementations in both **Python script files (`.py`)** and **Jupyter Notebook files (`.ipynb`)**. Some methods are coded in Python scripts for standalone execution, while others are demonstrated in notebooks for better visualization and step-by-step explanation. Be sure to check the appropriate file format based on your use case.
