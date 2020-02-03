# Musical Instrument Recognition

## Run locally

**Build**  
> `$ docker build -t bastdm5 .`

**Run**  
> `$ docker run -p 9002:9002 bastdm5`

=> go to http://localhost:9002


## Development

Run container and mount source code into the container:
> `$ docker run -it -p 9002:9002 -v /.../musical-instrument-recognition:/app bastdm5`

$ docker run -it -p 9002:9002 -v /Users/yast/workspace/hobby/musical-instrument-recognition:/app bastdm5

Create a virtual environment: 
> `$ virtualenv venv`

Activate virtual environment:
> `$ source venv/bin/activate`

Install dependencies:
> `$ pip install -r requirements.txt`

