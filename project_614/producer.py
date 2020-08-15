from __future__ import absolute_import
from kafka import KafkaProducer
import csv
import json

bootstrap_servers = ['localhost:9092']
topicName = 'numtest'
producer = KafkaProducer(bootstrap_servers = bootstrap_servers,max_request_size=4068518)
producer = KafkaProducer()

data = [producer.send(topicName, json.dumps(d).encode('utf-8')) for d in csv.DictReader(open('spambase.csv'))]

print("Data sent from Producer to Consumer")




