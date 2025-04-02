FROM python:3
WORKDIR /Phase 4 - Cloud Computing
COPY requirements_preprocessing.txt /Phase 4 - Cloud Computing
RUN pip install -r requirements_preprocessing.txt
COPY . /app
EXPOSE 5000
ENTRYPOINT ["python"]
CMD ["Phase 4 - Cloud Computing.py"]
