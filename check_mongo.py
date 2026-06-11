from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['telecom_churn_db']
docs = list(db['churn_forecasts'].find({}, {'_id': 0}))
print('Total docs:', len(docs))
if docs:
    print('Keys:', list(docs[0].keys()))
    print('Full doc:', docs[0])
else:
    print('Collection is EMPTY!')
