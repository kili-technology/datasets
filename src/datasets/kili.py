import json
import os
import requests

from kili.client import Kili
from tqdm import tqdm


def kili_decorator(func):
    def wrapper(*args, **kwargs):
        if len(args) == 0:
            raise ValueError('path must be specified')
        path = args[0]
        if path.startswith('kili/'):
            project_id = path.split('/')[-1]
            kili_api_key = kwargs.get('kili_api_key', '')
            kili = Kili(api_key=kili_api_key)
            total = kili.count_assets(project_id=project_id)
            first = 100
            assets = []
            for skip in tqdm(range(0, total, first)):
                assets += kili.assets(
                    project_id=project_id, 
                    first=first, 
                    skip=skip, 
                    disable_tqdm=True,
                    fields=[
                        'content',
                        'labels.createdAt',
                        'labels.jsonResponse', 
                        'labels.labelType'])
            assets = [{
                    **a,
                    'labels': [
                        l for l in sorted(a['labels'], key=lambda l: l['createdAt']) \
                            if l['labelType'] in ['DEFAULT', 'REVIEW']
                    ][-1:],
                } for a in assets]
            assets = [a for a in assets if len(a['labels']) > 0]
            data_files = os.path.join(
                os.getenv('HOME'), '.cache', 'huggingface', 'datasets', 'kili', f'{project_id}.json')
            with open(data_files, 'w') as handler:
                for asset in assets:
                    text = requests.get(asset['content'], headers={
                        'Authorization': f'X-API-Key: {kili_api_key}',
                    }).text
                    handler.write(json.dumps({
                        'text': text,
                        'label': asset['labels'][0]['jsonResponse'],
                    }) + '\n')
            return func('json', data_files=data_files)
        return func(*args, **kwargs)
    return wrapper