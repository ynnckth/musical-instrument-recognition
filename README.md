
Create a docker image with the tag "musical-instrument-prediction"
`docker build -t musical-instrument-prediction .`

Run a container mapping your machine's port 9002 to the container's exposed port 9002:
docker run -p 9002:9002 musical-instrument-prediction

List all running containers:
`docker ps`

SSH to container:
`docker exec -it <container_id> bash`

Show app logs:
`docker logs <container_id>`
