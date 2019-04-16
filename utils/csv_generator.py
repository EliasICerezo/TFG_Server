import pandas as pd 
import json 
import os

original_dir = "D:\Data_TFG\Images"
base_dir= "D:\Data_TFG\Images\images_new\Images"

def read_json_and_build_csv(csvname="dfnew_data.csv"):
    df = pd.DataFrame()
    print(df)
    descripttions = os.listdir("D:\Data_TFG\Images\images_new\Descriptions")
    
    for des in descripttions:
        with open(os.path.join(base_dir,des)) as file:
            data = json.load(file)
            try:
                row = [[str(os.path.join(base_dir,des)+".jpeg"),data["meta"]["clinical"]["benign_malignant"],data["meta"]["clinical"]["diagnosis"]]]
                df = df.append(row, ignore_index=True)
            except:
                pass
    #df.to_csv(save_path, index=False)
    print("Csv saved at location: "+csvname)
    df.to_csv(csvname,index=False)

def read_current_csv_and_modify(base_csv="D:\OneDrive\TFG\TFG_Python\HAM10000_metadata.csv", csvname = "juntoMod.csv"):
    df = pd.read_csv(base_csv)
    new_csv= pd.DataFrame()
    for i, row in df.iterrows():
        if pd.isna(row["path"]) :
            tipo = "benign"
            if row["dx"] == "mel":
                tipo = "malignant"
            if row["dx"] == "nv" or row["dx"] == "akiec":
                tipo = "pre-malignant"
            row["tipo"]=tipo
            row["path"]= os.path.join(original_dir, row["image_id"])

        if pd.isna(row["dx"]):
            path, file =  os.path.split(row["path"])
            row["path"] = os.path.join(base_dir,file)

        new_csv=new_csv.append(row, ignore_index=True)
    new_csv.to_csv(csvname,index=False)


def modify_new_csv(path):
    new_csv = pd.DataFrame()
    #TODO poner las nevus como premalignas
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        if not pd.isna(row["path"]):
            if row["diagnosis"] == "nevus":
                row["tipo"] = "pre-malignant"
                row["dx"] = "nv"
            if not pd.isna(row["diagnosis"]) and "melanoma" in row["diagnosis"]:
                row["dx"] = "mel"
        new_csv = new_csv.append(row,ignore_index=True)

    print(new_csv)
    return new_csv


def recortar():
    new_csv = pd.DataFrame()
    a = b = c =0
    df = pd.read_csv("D:\OneDrive\TFG\TFG_Python\dftest.csv")
    for i, row in df.iterrows():
        if row["tipo"] == "benign" and a < 2510:
            a+=1
            new_csv = new_csv.append(row,ignore_index=False)
        if row["tipo"] == "premalignant" and b < 2510:
            b+=1
            new_csv = new_csv.append(row, ignore_index=False)
        if row["tipo"] == "malignant" and c < 2510:
            c+=1
            new_csv = new_csv.append(row, ignore_index=False)
    new_csv.to_csv("dfrecortado.csv",index=False)


def modificalistas(paths, dircontent, dir):
    for p in paths:
        if p.startswith(dir):
            idx =paths.index(p)
            aux = os.path.basename(p)[:13]
            result = [i for i in dircontent if i.startswith(aux)]
            if len(result)==0 or len(result)>1:
                print("FALLO AL ENCONTRAR EL FICHERO: "+aux)
            paths[idx]=os.path.join(dir,result[0])
            print(paths[idx]+" , "+str(idx))

    return paths