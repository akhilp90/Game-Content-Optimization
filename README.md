# Game Level Recommendation System Using Player Behavior & Neural Collaborative Filtering (NCF)

## 📌 Overview

This project aims to enhance the user experience in games by recommending **appropriate levels** to new players. The system utilizes:

- **Behavioral analytics** to cluster players and levels
- **Machine learning models (KMeans)** for player segmentation and level difficulty classification
- **Deep learning (Neural Collaborative Filtering)** to personalize recommendations

By simulating a new user’s initial gameplay, the system predicts their skill level and recommends levels with the highest engagement potential for that skill class.

---

## Dataset Description

The dataset contains 10,000 gameplay records with 17 columns:

| Column                 | Description |
|------------------------|-------------|
| `player_id`            | Unique player identifier |
| `level_id`             | Unique game level identifier |
| `attempts`             | Number of attempts in a session |
| `success`              | 1 if player completed the level, else 0 |
| `moves_used`           | Number of moves used (some values missing) |
| `time_spent_sec`       | Total time spent in seconds |
| `used_booster`         | Whether player used booster items |
| `accuracy`             | Gameplay accuracy |
| `first_attempt_success`| Whether level was completed on first attempt |
| `num_retries`          | Number of retries for the level |
| `date_played`          | Date and time of play |
| `rage_quit`            | 1 if the player quit abruptly |
| `avg_time_per_attempt` | Average time per attempt |
| `num_obstacles`        | Number of obstacles in the level |
| `retry_gap_time`       | Time between retries |
| `session_id`           | Unique session ID |
| `avg_moves_per_attempt`| Average moves per attempt (some values missing) |

---

## Methodology

### 1.Player Skill Clustering

**Objective:** Classify players into three skill categories — Beginner, Intermediate, Expert — using behavioral statistics.

**Steps:**
- Grouped by `player_id` and computed average values for:
  - `attempts`, `success`, `first_attempt_success`, `time_spent_sec`
- Applied `StandardScaler` for normalization
- Used `KMeans (n_clusters=3)` for clustering
- Mapped clusters to human-readable skill levels using average stats:
  ```python
  {0: "Beginner", 1: "Intermediate", 2: "Expert"}
