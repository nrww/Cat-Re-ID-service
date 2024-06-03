import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Response
from typing import List
import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from PIL import Image

from oml.models import ViTExtractor
from oml.transforms.images.torchvision import get_normalisation_resize_torch
from ultralytics import YOLO
from data import Cat, CatData, Owner, Device, CatLog, Session
import data
import os
import matplotlib.pyplot as plt
import io
import json

class YOLOModel:
    def __init__(self, model_path, img_size=(480, 480), verbose=False):
        self.model = YOLO(model_path, verbose=verbose)
        self.img_size = img_size

    def segmentate_cat(self, imgs: List):
        results = self.model.predict(imgs, classes=[15], conf=0.5)
        ii = 0
        rez = []
        logs = []
        for r in results:
            ii+=1
            if len(r) == 0:
                logs.append('NF')
                continue
            if len(r) > 1:
                logs.append('OC')
                continue
            logs.append('OK')
            c = r[0]
            img = c.orig_img
            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
            mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask3ch, img)
            
            x1, y1, x2, y2 = c.boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
            iso_crop = isolated[y1:y2, x1:x2]
            
            rez.append(cv2.resize(iso_crop, self.img_size))
        if len(imgs) == 1:
            return rez[0], logs[0]
        return rez, logs

class FeatureExtractor:
    def __init__(self, model_path):
        self.extractor = ViTExtractor(model_path, arch="vits16", normalise_features=False)
        self.transform = get_normalisation_resize_torch(im_size=224)
        
    @torch.no_grad()
    def cat_to_vec(self, img):
        if not isinstance(img, list):
            img = [img,]
        imgs = []
        for i in img:
            i = Image.fromarray(i)
            i = self.transform(i)
            imgs.append(i)
        vec = self.extractor(torch.stack(imgs)).numpy()
        return vec
    
class NearestNeighborsModel:
    def __init__(self, vec_data_path):
       
        with Session() as session: 
            temp = session.query(Cat.id, CatData.vector).join(CatData).all()
            temp = np.array(temp)
            self.labels = temp[:,0].astype(int)
            self.vec = list(map(lambda x: np.frombuffer(x, dtype=np.float32), temp[:,1]))
            self.vec = np.array(self.vec)
                
        self.knn = NearestNeighbors(algorithm="auto", p=2, n_jobs=-1)
        self.knn.fit(self.vec)
        
    def update_knn(self, vec, cat_id):
        # Исправить
        if len(vec.shape) < 2:
            vec = np.expand_dims(vec, axis=0)
        self.vec = np.concatenate((self.vec, vec))
        self.labels = np.concatenate((self.labels, 
                                      np.full(vec.shape[0], cat_id)))
        self.knn.fit(self.vec)
        
    @staticmethod
    def str_to_float_list(x):
        x = x.split(', ')
        x = list(map(float, x))
        return x

class CatDetectorApp:
    def __init__(self, top_k=10):
        self.app = FastAPI()
        self.yolo_model = YOLOModel('yolov8x-seg.pt')
        self.feature_extractor = FeatureExtractor("vits16_cat.ckpt")
        self.nn_model = NearestNeighborsModel('val_vec.pd')
        self.top_k = top_k
        
        self.knn = self.nn_model.knn

    def run(self):
        @self.app.post("/add_cat/")
        def add_cat(files: List[UploadFile] = File(...), 
                    cat_name: str = '', owner: str = ''):
            
            path = f'img/{str(owner)}/{cat_name}'
            imgs = []
            try:

                shift = 1
                if not os.path.exists(path):
                    os.makedirs(path)
                else:
                    shift = max(map( lambda x: int(x.strip('.jpg').split('_')[-1]) ,
                            os.listdir(path))) + 1

                for file in files:
                    contents = file.file.read()
                    img = np.frombuffer(contents, np.uint8)
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)               
                    imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


                seg_cats, logs = self.yolo_model.segmentate_cat(imgs)
                if len(seg_cats) == 0:
                    return {'cat_id' :  'none', 'n_add_img': len(seg_cats) ,'n_all_img': len(seg_cats) + shift - 1, 'error': logs}
                vecs = self.feature_extractor.cat_to_vec(seg_cats)

                paths = []
                for i, img in enumerate(seg_cats):
                    file_path = f'{path}/{cat_name}_{str(i + shift)}.jpg'
                    paths.append(file_path)
                    cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                catID = data.cat_to_db(vecs, paths, cat_name, owner, is_pet=True)

                self.nn_model.update_knn(vecs, catID)
            except Exception as inst:
                print(inst) 
                return {"Error": str(inst)}
            finally:
                file.file.close()
            return {'cat_id' :  catID, 'n_add_img': len(seg_cats) ,'n_all_img': len(seg_cats) + shift - 1, 'error': logs}

        @self.app.post("/check_cat/")
        def check_cat(file: UploadFile = File(...), owner: str = '', conf: float = 0.5, debug: bool = True):
            import time
            elapsed_time = time.time()
            response = {'pass': '0'}
            try:
                contents = file.file.read()
                img = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                seg_cat, _ = self.yolo_model.segmentate_cat(img)

                if len(seg_cat) == 0:
                        print({'pass': '0'})
                        return {'pass': '0'}

                vec_cat = self.feature_extractor.cat_to_vec(seg_cat)

                dists, ii_closest = self.knn.kneighbors(vec_cat, n_neighbors=self.top_k, return_distance=True)
                ii_closest += 1
                with Session() as session: 
                    owner_cat_list = session.query(Owner.id, Cat.id, Cat.is_pet, Cat.name).join(Cat).filter(Owner.id == owner).all()

                    paths_closest = []
                    catID_closest = []
                    name_closest = []
                    for i in ii_closest[0]:
                        q = session.query(CatData.img, Cat.id, Cat.name).join(Cat).filter(CatData.id == i).first()
                        paths_closest.append(q[0])
                        catID_closest.append(q[1])
                        name_closest.append(q[2])
                cat_info = {}
                for i in owner_cat_list:
                    catID = i[1] 
                    fc = np.count_nonzero(np.array(catID_closest) == catID)
                    ac = np.count_nonzero(self.nn_model.labels == catID)
                    prec = fc / min(ac, self.top_k)
                    print('catdataclos',ii_closest)
                    print('catidclos',catID_closest)
                    print(fc, ac, prec, catID)
                    cat_info[catID] = prec

                    if prec > conf or response['pass']=='1':
                        response['pass'] = '1'

                from operator import itemgetter
                closest_cat = max(cat_info.items(),key=itemgetter(1))
                response['cat_id'], response['conf'] = map(str, closest_cat)
                response['nearest_cats'] = json.dumps(catID_closest)
                px = 1/plt.rcParams['figure.dpi']
                fig,ax = plt.subplots(1, self.top_k, figsize=(2500*px,500*px))
                print(name_closest)
                for j, jj in enumerate(paths_closest):
                    img = cv2.imread(jj)
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    ax[j].imshow(img)
                    ax[j].set_title(name_closest[j])
                    ax[j].axis('off')

                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                
            except Exception as inst:
                print(inst) 
                return {"error": str(inst)}
            finally:
                file.file.close()

            elapsed_time = int((time.time() - elapsed_time)*1000)
            response['time'] = str(elapsed_time) 
            print(f"time: {elapsed_time} ms")
            if debug:
                return Response(content=buf.getvalue(), headers=response, media_type="image/png")
            else:
                return {'pass': response['pass']}

        import uvicorn
        uvicorn.run(self.app, host="0.0.0.0", port=8080)

if __name__ == "__main__":
    app = CatDetectorApp()
    app.run()
