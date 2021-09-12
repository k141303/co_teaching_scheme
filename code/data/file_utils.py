import os
import tqdm
import json

from multiprocessing import Pool
import multiprocessing as multi

class FileUtils:
    class File:
        def load(file_path):
            with open(file_path, "r") as f:
                return f.read()

    class JsonL:
        def load(file_path):
            data = []
            with open(file_path, "r") as f, \
                tqdm.tqdm(desc=os.path.basename(file_path)) as t:
                for d in map(json.loads, f):
                    data.append(d)
                    t.update()
            return data

        def mp_load(file_path):
            data = []
            with open(file_path, "r") as f, \
                Pool(multi.cpu_count()) as p, \
                tqdm.tqdm(desc=os.path.basename(file_path)) as t:
                for d in p.imap(json.loads, f):
                    data.append(d)
                    t.update()
            return data

        def save(file_path, data):
            with open(file_path, "w") as f:
                dumps = map(json.dumps, data)
                f.write("\n".join(dumps))

    class Json:
        def save(file_path, data):
            with open(file_path, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True, separators=(',', ': '))

        def load(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
