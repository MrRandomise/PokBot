import json


def ReadJson(file):

    with open(file) as f:
        templates = json.load(f)
    return templates


def GetRectangle(templates):
    mas = []
    for Rectangle in templates:
        mas.append([Rectangle["PosX"], Rectangle["PosY"], Rectangle["Width"], Rectangle["Heigth"], Rectangle["Name"]])
    return mas
