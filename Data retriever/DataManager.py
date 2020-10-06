import json

import pandas as pd


def saveData(dataList, fileName, format="csv"):
    def saveAsJson(data, fileName):
        with open(fileName + ".txt", "w") as outfile:
            json.dump(data, outfile)

    def saveAsCsv(data, fileName):
        df = pd.read_json(data)
        df.to_csv(r"../Datasets/" + fileName + ".csv", index=None)

    json_format = json.dumps(dataList)
    if format == "csv":
        saveAsCsv(json_format, fileName)
    elif format == "json":
        saveAsJson(json_format, fileName)
    else:
        raise Exception("Invalid save format given")
    print("Saved data successfully to file " + fileName + " in format: " + format)