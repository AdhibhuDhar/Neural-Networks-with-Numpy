import json
import datetime
class ExperimentLogger:
    def __init__(self,filepath="experiments.json"):
        self.filepath=filepath
    def log(self,config,results):
        entry={
            "timestamp":str(datetime.datetime.now()),
            "config":config,
            "results":results
        }
        try:
            with open(self.filepath,"r") as f:
                data=json.load(f)
        except:
            data=[]
        data.append(entry)
        with open(self.filepath,"w") as f:
            json.dump(data,f,indent=4)