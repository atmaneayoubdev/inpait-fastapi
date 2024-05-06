# 
FROM python:3.11

# 
WORKDIR /code

# Install necessary system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]