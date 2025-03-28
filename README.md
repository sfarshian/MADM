# MADM 🚀  

Have you ever tried to compare a few options and gotten stuck? 🤔  

The choices seem logical, but you just can't decide on the best one? 😅  

I get it—**it takes Math to choose!** 🔢✨  

## What is this? 🤷‍♂️  

This is a simple implementation of well-known methods for **multi-attribute decision making (MADM)**. Whether you're making everyday decisions or tackling larger ones (like buying a house 🏡), this repo can help you choose the best option with confidence!  

## What It Can Do 🔍  

### Matrix Preprocessing 🔄  

Before making a final decision, you need to **normalize** your decision matrix to avoid mixing different metrics together. This code helps you with that:  

✅ **Normalizes** the decision matrix using three different methods:  
   ✅ **Linear Normalization** 🔄  
   ✅ **Euclidean Normalization** 🧮  
   ✅ **Max-Min Normalization** 📊  

### Methods 🛠️  

Once the matrix is normalized, there are several ways to calculate the best alternative. Here’s a brief overview:  

#### 1. **Permutation Method** 🔀  
This method compares alternatives by evaluating their pairwise concordance and net scores. While effective, it’s computationally expensive. We **strongly advise** not using more than **5 attributes** with this method. If you have more than 5, consider one of the other methods.  

✅ Implements the **Permutation Method** to calculate:  
   ✅ **Concordance matrices** 📈  
   ✅ **Net scores** for different permutations of alternatives 🔢  

#### 2. **TOPSIS Method** ⭐  
The **TOPSIS** method ranks alternatives based on their closeness to the ideal solution (and distance from the worst solution). It's a great choice for balanced decision-making!  

✅ Implements the **TOPSIS Method** to rank alternatives based on proximity to the ideal solution 🏆  

#### 3. **Simple Additive Weighting (SAW)** ➕  
If you prefer a more straightforward approach, the **SAW** method is one of the most intuitive decision-making techniques. It scores alternatives by summing their weighted normalized values.  

✅ Implements the **SAW Method**, which scores alternatives by summing their weighted normalized values ⚖️  

#### 4. **ELECTRE I Method** 🛑  
For a more rigorous decision-making process, the **ELECTRE I** method eliminates weaker alternatives using concordance and discordance indices.  

✅ Implements the **ELECTRE I Method** to compare alternatives based on concordance and discordance indices ⚔️  

#### 5. **Save Results to File** 💾  
Want to save your results for analysis? We've got you covered! All results are outputted into a text file for easy access and further analysis.  

✅ Outputs results into a **text file** for easy analysis 📑  

### How These Methods Work 🔧  

Each of these methods follows a structured approach:  

1️⃣ **Normalization**: Transform raw values into a comparable scale ⚖️.  
2️⃣ **Weighting**: Assign importance levels to different criteria 🎯.  
3️⃣ **Calculation**:  
   ✅ **SAW**: Sum the weighted scores for each alternative.  
   ✅ **TOPSIS**: Measure distances from the ideal and worst solutions.  
   ✅ **Permutation Method**: Compare alternatives using pairwise concordance and net scores.  
   ✅ **ELECTRE I**: Use concordance and discordance thresholds to eliminate weaker choices.  
4️⃣ **Ranking**: Identify the best alternative based on the chosen method 📊.  

Each method has its strengths:  
✅ **SAW** is simple but effective  
✅ **TOPSIS** balances ideal and worst-case scenarios  
✅ **Permutation Method** provides a thorough comparison  
✅ **ELECTRE I** is great for eliminating poor options  

## Prerequisites 📋  

To get started, you’ll need the following:  

✅ Python 🐍  
✅ `pandas` library 📚  
✅ `numpy` library ➗  

## Disclaimer ⚠️  

This project includes both **Python script files (`.py`)** and **Jupyter Notebook files (`.ipynb`)**. Some methods are implemented in Python scripts for standalone execution, while others are demonstrated in notebooks for better visualization and step-by-step explanation. Make sure to check the appropriate file format based on your needs.  
