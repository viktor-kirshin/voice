from transformers import pipeline

classifier = pipeline(model="seara/rubert-tiny2-russian-sentiment")

result = classifier("")
print(result)