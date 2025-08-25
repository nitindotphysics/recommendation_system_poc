# Recommendation System Proof of Concept
A comprehensive proof-of-concept recommendation system implementing multiple algorithms with an interactive web interface using gradio.

## Overview
This project demonstrates four fundamental recommendation algorithms:

- **Content-based Recommendation Algorithm:** Recommends items similar to those a user has liked in the past
- **User-User Collaborative Filtering:** Finds similar users and recommends items they've liked
- **Item-Item Collaborative Filtering:** Finds similar items based on user preferences
- **Singular Value Decomposition (SVD):** Matrix factorization technique for dimensionality reduction

## Quick Start
Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation & Setup
```bash
# Clone the repository
git clone https://github.com/nitindotphysics/recommendation_system_poc.git
cd recommendation_system_poc

# Create and activate a virtual environment

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate```

# For Windows
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Note: This may take several minutes depending on your network connection
```

### Run the application

```bash
python3 app.py # on macOS or Linux
python app.py # on windows
```

### Access the application
- Copy the URL generated in the terminal
- Paste it into your web browser

## Algorithms Implemented
1. **Content-based Alogorithm**
Uses item features to recommend similar items based on a user's preferences.

2. **User-User Collaborative Filtering**
Finds users with similar tastes and recommends items that these similar users have liked.

3. **Item-Item Collaborative Filtering**
Identifies similar items based on how users have rated them and recommends items similar to those a user has liked.

4. **Singular Value Decomposition (SVD)**
A matrix factorization technique that reduces dimensionality to discover latent features in the user-item interaction matrix.

## Dataset
The application uses the MovieLens 100K dataset containing:

- User ratings
- Movie metadata
- User information

## Usage
Launch the application following the installation steps

Select a recommendation algorithm from the sidebar

Input user ID

Adjust parameters as needed

View the generated recommendations

### Troubleshooting
Common Issues
Python not found error

```bash
# Use python3 instead of python on some systems
python3 -m venv venv
python3 app.py
```
### Network issues during installation 
Check your network connection

### Acknowledgments
MovieLens for providing the dataset

The recommender systems research community

Gradio for the web framework