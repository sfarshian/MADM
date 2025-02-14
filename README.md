# MADM

Have you ever tried to compare a couple of things and got stuck?

The options seem logical, but you can’t choose the final one?

I know, it takes Math to choose!

## What is this

This is a simple implementation of well-known methods for **multi-attribute decision making (MADM)**. You can use it in your day-to-day tasks or for bigger decisions such as buying a house.

## What it can do

Before making a final decision, you need to normalize your decision matrix to avoid having multiple metrics mixed together. This code can help you with that:

- **Normalizes** decision matrix using three different methods:
  - **Linear Normalization**
  - **Euclidean Normalization**
  - **Max-Min Normalization**
  
After normalization, you have multiple ways to find your best option. One of them is the Permutation Method, which is effective but computationally expensive. We strongly advise not using more than 5 attributes with this method—if you have more than 5, consider using one of the other methods.

- Implements the **Permutation Method** to calculate:
  - Concordance matrices
  - Net scores for different permutations of alternatives

Then, there’s the good old TOPSIS Method which ranks alternatives based on their closeness to ideal solutions:

- Implements the **TOPSIS Method** to rank alternatives based on closeness to ideal solutions.

If you’re a data enthusiast and want your results and calculations saved, we’ve got you covered! All results are outputted into a text file for easy analysis.

- Outputs results into a text file for easy analysis.

## What you need beforehand

- Python 3.x
- `pandas` library
- `numpy` library

## Let me show you
<!-- Example -->