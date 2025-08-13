from datasets import load_dataset

ds = load_dataset('json', data_files={
        'dev': f'dataset/medmcqa/dev.json',
    })
print(ds)