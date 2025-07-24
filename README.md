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

### 🎯 Objective

Recommend game levels that align with a player's skill and behavioral patterns using deep learning-based collaborative filtering.

---

### 🧠 What is NCF?

Neural Collaborative Filtering (NCF) is a deep learning-based recommendation algorithm that learns non-linear interaction patterns between users (players) and items (game levels). Unlike traditional matrix factorization, NCF uses neural networks to capture complex user-item interactions.

---

### ⚙️ How Our NCF Works

#### 1. **Framing the Problem**

* **Type**: Multiclass recommendation task.
* **Goal**: Predict the engagement quality between a player and a level, which is later used to recommend the most appropriate levels.
* **Labeling**:

  * **0** → Easy win (completed quickly)
  * **1** → Ideal engagement (balanced effort and success)
  * **2** → Rage quit / Overwhelmed

This label acts as a **proxy signal** to help the model rank levels based on engagement suitability rather than pure classification.

---

#### 2. **Feature Preparation**

* **Categorical Inputs**:

  * `player_id` → Label encoded
  * `level_id` → Label encoded

* **Engagement Labeling Rule**:

  * Label is determined using rules based on `success`, `rage_quit`, `attempts`, and `time_spent_sec`

---

#### 3. **Model Architecture**

* **Embedding Layers**:

  * `player_id` and `level_id` are each passed through embedding layers to learn dense vector representations.

* **Concatenation**:

  * Player and level embeddings are concatenated.

* **Neural Network**:

  * Dense Layer 1 → ReLU activation → Dropout
  * Dense Layer 2 → ReLU activation → Dropout
  * Output Layer → 3 logits (softmax for class probabilities: \[easy win, ideal, rage quit])

---

#### 4. **Training Setup**

* **Loss Function**: CrossEntropyLoss (because of multi-class labels)
* **Optimizer**: Adam
* **Evaluation Metric**: Loss during training; precision and coverage used for recommendations

---

#### 5. **Cold Start Handling for New Players**

1. A new player plays 2–3 tutorial levels.
2. Their gameplay is recorded (`success`, `time_spent`, `retries`, etc.).
3. Their **engagement label** is predicted using rules.
4. Based on mode of those labels, their **inferred skill** (Beginner/Intermediate/Advanced) is derived.
5. Average player embedding of users with similar skill class is computed.
6. This embedding is assigned to the new user.
7. The model scores all levels for this user.
8. Top-N levels with highest `ideal engagement` score (class `1`) are recommended.

---

### 📈 Example Flow for New User

```python
simulate_gameplay(new_player)
→ collect_stats()
→ infer_engagement_label()
→ assign_skill_class()
→ average_similar_user_embeddings(skill_class)
→ embed_new_user()
→ for each level:
    predict_engagement_score()
→ sort levels by class-1 probability
→ return top_k_levels()
```

This approach ensures that recommendations are not just based on difficulty but on what historically led to **meaningful engagement** for similar players.

---

Let me know if you want the evaluation section converted next or visual aids added to the README!



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

## 7. 🤖 Reinforcement Learning Agent for Adaptive Level Recommendation

### 🎯 Objective

To further personalize level recommendations, we implement a **Deep Q-Network (DQN)**-based **reinforcement learning (RL)** agent that dynamically learns which levels to recommend based on a player’s ongoing performance and behavior over time.

This goes beyond static recommendations — now the system **learns from player interaction**, updating its strategy to keep engagement high and frustration low.

---

### 🧠 Environment Setup

We define a custom environment `CandyCrushEnv` that simulates:

- Player skill changes over time
- Level difficulty
- Win probability as a function of skill and level
- Behavioral factors like retries, boosters used, stars earned, and time spent

Each **state** includes:

- `player_skill`: Current skill level of the player
- `level_difficulty`: The difficulty of the selected level
- `boosters_used`: Whether boosters were used
- `win_prob`: System’s predicted win probability
- `num_retries`: Number of retries taken
- `avg_time`: Average time spent on the level
- `stars_earned`: Stars earned (1–3)
- `level_normalized`: Normalized ID of the level

The **action** is choosing a level to recommend.

---

### 🧮 Reward Function

The reward is carefully shaped to balance success, challenge, and engagement:

```python
reward = (
    3 * stars_earned +        # Bonus for engagement
    2 * int(level_completed) -# Bonus for completion
    0.7 * num_retries -       # Penalty for too many retries
    0.5 * boosters_used -     # Penalty for over-reliance on boosters
    0.1 * (avg_time / 60)     # Mild penalty for long session time
)
```  

 ### 🧱 DQN Architecture
A deep neural network is used as the Q-function approximator. Architecture:

Input: 8-dimensional state vector

Hidden Layers: [256 → 128 → 64] with ReLU activations

Output: Q-values for 50 possible levels

Uses Double DQN to reduce overestimation bias.

### ⚙️ Agent Mechanics
The RL agent handles:

Action Selection: ε-greedy strategy to balance exploration & exploitation

Experience Replay: Stores gameplay samples to train on batches

Target Network: A slowly updated copy of the main model for stability

Training Loop: Backpropagates Bellman error and updates Q-values

Key Hyperparameters:

γ = 0.97 (discount factor)

ε = 1.0 → 0.01 (exploration decay)

LR = 0.0005, batch_size = 64

### 📊 Training & Evaluation
The agent is trained over 500 episodes, simulating player sessions. Key plots:

📈 Rewards per Episode: Indicates whether agent is recommending better over time

🔻 Epsilon Decay: Shows the agent learning to exploit more over time

Example training output:

python-repl
Copy
Edit
Episode 50/500 - Reward: 12.5 - Epsilon: 0.6043
...
Episode 500/500 - Reward: 24.3 - Epsilon: 0.0100
### 🧠 Why RL?
While NCF personalizes based on historical data, RL dynamically adapts in real-time to a player's skill changes and reactions to previous levels.

📊 NCF = Offline prediction based on past interactions

🤖 RL = Online decision-making based on sequential player behavior

Together, they provide both cold-start personalization and adaptive session flow.

### 🛠 Future Improvements
Integrate real gameplay telemetry (from actual player logs)

Use more complex environments (multi-step level chains, special events)

Integrate player mood/emotion estimation as part of state

Implement Multi-Armed Bandit models for faster deployment decisions




