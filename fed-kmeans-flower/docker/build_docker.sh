if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "Using PAT ending in: ${GIT_PAT: -4}"
docker build --build-arg GIT_PAT=$GIT_PAT -t fed-kmeans -f Dockerfile ..