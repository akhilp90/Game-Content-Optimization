# 🎮 Game Level Recommendation System using Player Behavior & Neural Collaborative Filtering (NCF)

## 📌 Overview

This project aims to personalize the **game level recommendation** experience for players based on their behavior and skill level.

The system works in **three stages**:

1. **Player Skill Clustering** using behavioral statistics
2. **Level Difficulty Classification**
3. **Personalized Recommendations** using Deep Learning (Neural Collaborative Filtering)

The goal is to simulate a **cold start** scenario for a new player and recommend levels that match their skill and maximize engagement.

---

## 📂 Dataset Description

The dataset consists of 10,000 gameplay records with the following structure:

| Column Name             | Description                                         |
|------------------------|-----------------------------------------------------|
| `player_id`            | Unique player identifier                            |
| `level_id`             | Unique level identifier                             |
| `attempts`             | Total attempts in one session                       |
| `success`              | 1 if the level was completed, else 0                |
| `moves_used`           | Number of moves used (some missing values)          |
| `time_spent_sec`       | Time spent on the level in seconds                  |
| `used_booster`         | Whether the player used booster items               |
| `accuracy`             | Accuracy during gameplay                            |
| `first_attempt_success`| 1 if level cleared on first try                     |
| `num_retries`          | Number of retries                                   |
| `date_played`          | Timestamp of gameplay                               |
| `rage_quit`            | 1 if the player quit suddenly                       |
| `avg_time_per_attempt` | Average time per attempt                            |
| `num_obstacles`        | Number of obstacles in the level                    |
| `retry_gap_time`       | Time between retries                                |
| `session_id`           | Unique session identifier                           |
| `avg_moves_per_attempt`| Average moves per attempt (some missing values)     |

---

## 🔍 Methodology

## 1. 🧠 Player Skill Clustering 
**Goal: Group players into skill categories based on how they interact with game levels.**

How it works:

We first aggregate gameplay behavior for each player by computing averages of key metrics like number of attempts, success rate, first-attempt success rate, and time spent.

These aggregated features are then normalized using standard scaling, so no single feature dominates the clustering process due to scale.

We use KMeans clustering with 3 clusters, assuming players generally fall into three intuitive skill levels: Beginner, Intermediate, and Expert.

Once clustered, we analyze the average stats per cluster to interpret and assign each cluster a human-readable label. For instance:

Beginners typically have low success rates, more retries, and longer time spent.

Experts succeed quickly, often on the first attempt, and spend less time per level.

## 2. 🎮 Level Difficulty Classification
**Goal: Determine how hard each level is, based on aggregated player performance.**

How it works:

We aggregate player-level interactions for each level: success rates, average attempts, average time spent, retry counts, and rage quits.

Similar to player clustering, we apply KMeans (with 3 clusters) to categorize levels into Easy, Medium, and Hard.

The clusters are labeled based on statistical characteristics:

Easy levels have high completion and first-attempt success rates.

Hard levels are often rage-quit and require more retries and time.

## 3. 🧩 Feature Engineering
**Goal: Prepare a clean dataset for collaborative filtering.**

Steps:

Merged the player skill clusters and level difficulty clusters into the main dataset.

Cleaned missing values in moves_used and avg_moves_per_attempt using mean imputation.

Transformed categorical features like boosters used and rage quit into binary form.

Final dataset columns include player_id, level_id, player_skill, level_difficulty, and engagement features.

## 4. 🔁 Neural Collaborative Filtering (NCF) – Personalized Recommendation
**Goal: Recommend levels tailored to a specific player's inferred skill and preferences.**

How it works:

We frame the problem as a binary recommendation task, where the model learns to predict whether a player would successfully engage with a level.

Player and level IDs are encoded and passed through embedding layers.

A neural network is trained on these embeddings along with interaction features like skill level, difficulty level, retries, boosters, etc.

The model learns latent patterns in how different player profiles interact with levels.

For a new user:

We simulate a few early games.

Based on those outcomes, the player is clustered into a skill level.

The model then recommends levels that had the highest success and engagement scores for similar players.

## 5. 🧪 Evaluation & Results
Evaluation Metrics:

Precision@K: How many of the top K recommended levels were actually suitable.

Coverage: Diversity of levels being recommended across players.

Hit Ratio: Whether at least one successful level is in the top-K predictions.

Observations:

High precision indicates the model recommends well-suited levels.

Good coverage ensures the system isn't biased toward a small subset of levels.

Personalized recommendations outperform random or popularity-based baselines.

## 6. 🧠 Why This Works
Traditional recommender systems don’t account for player learning curves or skill progression.

Our system blends unsupervised clustering with deep learning to adaptively recommend levels, especially useful in dynamic gaming environments where content is frequently updated.

Even new users can be accurately served after just a few rounds of gameplay.




