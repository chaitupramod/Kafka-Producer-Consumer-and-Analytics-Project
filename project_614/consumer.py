from __future__ import absolute_import
from kafka import KafkaConsumer
import sys
import json
import sys
import pandas as pd



bootstrap_servers = ['localhost:9092']
topicName = 'numtest'
consumer = KafkaConsumer (topicName, group_id = 'group1',bootstrap_servers = bootstrap_servers,
auto_offset_reset = 'latest', value_deserializer=lambda x: json.loads(x.decode('utf-8')))

dataList = []
header_list = []

try:
    print("IN CONSUMER")
    counter=1
    for message in consumer:
        dataList.append(list(dict(message.value).values()))
        #print(message.value)
        if(counter==4601):
            header_list.append(list(dict(message.value).keys()))
            break
        counter=counter+1
    print("BROKEN")
    
    print("*"*100)

    df = pd.DataFrame(dataList)
    print(df)

    print(header_list[0])

    
    df = df.astype(float)
    df.columns = header_list[0]
    
    print("Recived the data from the producer")

    df.to_csv("spambase_test_new.csv",index=False)
    
    from optional_proj_614 import analysis
    analysis()

    #call_analytics_program()

    



except KeyboardInterrupt:
    sys.exit()