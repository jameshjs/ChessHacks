import kagglehub

# Download latest version
path = kagglehub.dataset_download("anthonytherrien/leela-chess-zero-self-play-chess-games-dataset-3")

print("Path to dataset files:", path)
