from kafka import KafkaConsumer
import json
import pymongo
from app import speaker
from os.path import join
import pymongo
from utils import save_content

URI="text"
client = pymongo.MongoClient(URI)
db=client['text']
audios_folder="audios"
content_folder='test-cases'
consumer = KafkaConsumer(bootstrap_servers = 'text',
                        group_id= 'text',
                        auto_offset_reset =  'text',
                        security_protocol =  'text',
                        sasl_mechanism = 'text',
                        sasl_plain_username  =  'text',
                        sasl_plain_password  =  'text',
)

consumer.subscribe(["text"])

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
    
