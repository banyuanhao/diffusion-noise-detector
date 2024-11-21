import json
with open('prompts.json') as f:
    data = json.load(f)
    data = data['animals_objects']
    set_ani = set()
    set_obj = set()
    for i in data:
        set_ani.add(i.split(' ')[1])
        set_obj.add(i.split(' ')[-1])
    print(set_ani)
    print(set_obj)