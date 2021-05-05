import json
import hashlib

class CFakeHP:
  def __init__(self, values):
    self._values = values
    return
  
  def __getattr__(self, _):
    def wrapper(name, *a, **kw):
      return self._values[name]
    return wrapper
#######
def saveTrial(trial, saveTo):
  config = trial.get_config()
  values = {k: v for k, v in config['values'].items() if not k.startswith('tuner/')}
  values = json.dumps(values, indent=2, sort_keys=False)
  with open(saveTo(hashlib.md5(values.encode('utf8')).hexdigest()), 'w') as f:
    f.write(values)
    
  return

def loadTrial(filename):
  with open(filename, 'r') as f:
    values = json.load(f)
    
  return CFakeHP(values)
