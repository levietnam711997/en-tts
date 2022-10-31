from kafka import KafkaConsumer
import json
import pymongo
from app import speaker
from os.path import join
import pymongo
from utils import save_content

URI="mongodb://test-ai:Hahalolo%402022@10.10.12.201:27017,10.10.12.202:27017,10.10.12.203:27017/?replicaSet=test&authSource=test-ai"
client = pymongo.MongoClient(URI)
db=client['test-ai']
audios_folder="audios"
content_folder='test-cases'
consumer = KafkaConsumer(bootstrap_servers = '10.10.15.72:9092,10.10.15.73:9092,10.10.15.74:9092',
                        group_id= 'social_consumer_namlv',
                        auto_offset_reset =  'latest',
                        security_protocol =  'SASL_PLAINTEXT',
                        sasl_mechanism = 'SCRAM-SHA-512',
                        sasl_plain_username  =  'admin-hahalolo',
                        sasl_plain_password  =  'Hahalolo@2021',
)

consumer.subscribe(["topic_social_post.test-api-social.post"])

for message in consumer:
    
    document = json.loads(message.value.decode("utf-8"))['fullDocument']
    try:
        if document['typpost'] not in ['nor','exp']:
            continue
        _id=document['_id']['$oid']
        # print(document)
        print(document)
        lang=document['lang']
        onwer=document['owner']
        if 'content' in document and lang=="en":
            content=document['content']
            audio_filename=join(audios_folder,f'{_id}.wav')
            speaker(content,audio_filename)
            db['tts-dataset'].insert_one({
                'post_id':_id,
                'user_id':onwer,
                'content':content,
                'lang':lang,
                'audio_filename':audio_filename
            })
            save_content(f'{content_folder}/{_id}.txt',content)
            
    except:
        pass
    