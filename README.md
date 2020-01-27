# Musical Instrument Recognition

## Run locally

**Build**  
> `$ docker build -t bastdm5 .`

**Run**  
> `$ docker run -p 9002:9002 bastdm5`

=> go to http://localhost:9002


## Development

Create a virtual environment: 
> `$ virtualenv venv`

Activate virtual environment:
> `$ source venv/bin/activate`

Install dependencies:
> `$ pip install -r requirements.txt`

Optional: set the project interpreter of your IDE to the virtual environment