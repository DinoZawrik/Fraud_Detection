.PHONY: train app test lint docker

train:
	python main.py

app:
	streamlit run app.py

test:
	python -m pytest tests/ -v

lint:
	ruff check .

docker:
	docker build -t fraud-detection . && docker run -p 8501:8501 --rm fraud-detection
