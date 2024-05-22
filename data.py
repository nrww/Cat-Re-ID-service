from sqlalchemy import create_engine, MetaData, Table, Integer, String, \
    Column, DateTime, ForeignKey, Numeric, Boolean, LargeBinary
from datetime import datetime
from sqlalchemy.orm import sessionmaker, registry
import pandas as pd
import pymysql.cursors
import os

DB_HOST = os.environ["DB_HOST"]
DB_PORT = os.environ["DB_PORT"]
DB_LOGIN = os.environ["DB_LOGIN"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_DATABASE = os.environ["DB_DATABASE"]

conn = pymysql.connect(host=DB_HOST,
                       port=DB_PORT,
                       user=DB_LOGIN,
                       password=DB_PASSWORD)

with conn:
    with conn.cursor() as cursor:
        sql = f"CREATE DATABASE IF NOT EXISTS {DB_DATABASE} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        cursor.execute(sql)

    conn.commit()

engine = create_engine(f"mysql+pymysql://{DB_LOGIN}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_DATABASE}?charset=utf8mb4")
Session = sessionmaker(bind=engine)
metadata = MetaData()
mapper_registry = registry()

def init_data(vec_data_path='val_vec.pd'):
    
    with Session() as session:
        if session.query(Cat).count() == 0:
            owner = Owner(chat_id = 'default')
            session.add(owner)
            session.commit()
            df = pd.read_pickle(vec_data_path)
            df = df.fillna('none')
            last_label = -1
            for i, row in df.iterrows():
                label = row['label']
                if last_label != label:
                    last_label = label
                    cat = Cat(name = str(row['name']), owner_id=owner.id, is_pet = False)
                    session.add(cat)
                    session.commit()
                cat_data = CatData(vector = row['vec'].tobytes(), img=row['path'], cat_id=cat.id)
                session.add(cat_data)
            session.commit()


def cat_to_db(vecs, paths, name, owner, is_pet):

    with Session() as session:
        owner = session.query(Owner.id).filter(Owner.chat_id == owner).first()
        if owner is None:
            raise 'Owner not found'
        cat = session.query(Cat).join(Owner).filter(Owner.chat_id == owner, Cat.name == str(name)).first()
        if cat is None:
            cat = Cat(name = str(name), owner_id=owner.id, is_pet = is_pet)
            session.add(cat)
            session.commit()
        for vec, path in zip(vecs, paths):
            cat_data = CatData(vector = vec.tobytes(), img=path, cat_id=cat.id)
            session.add(cat_data)
        session.commit() 
        return cat.id
            

owner = Table('owner', metadata,
    Column('id', Integer(), primary_key=True),
    Column('chat_id', String(100), nullable=False),
    Column('reg_date', DateTime(), default=datetime.now),
)


cat = Table('cat', metadata,
    Column('id', Integer(), primary_key=True),
    Column('name', String(4000), nullable=False),
    Column('owner_id', ForeignKey('owner.id')),
    Column('is_pet', Boolean(), nullable=False),
)


cat_data = Table('cat_data', metadata,
    Column('id', Integer(), primary_key=True),
    Column('vector', LargeBinary(4000), nullable=False),
    Column('cat_id', ForeignKey('cat.id')),
    Column('img', String(4000)),
)


device = Table('device', metadata,
    Column('id', Integer(), primary_key=True),
    Column('name', String(200), nullable=False),
    Column('create_date', DateTime(), default=datetime.now),
    Column('owner_id', ForeignKey('owner.id')),
)


cat_log = Table('cat_log', metadata,
    Column('id', Integer(), primary_key=True),
    Column('date', DateTime(), default=datetime.now),
    Column('cat_id', ForeignKey('cat.id')),
    Column('device_id', ForeignKey('device.id')),
    Column('vector', String(4000), nullable=False),
    Column('is_openned', Boolean(), nullable=False),
    Column('has_entered', Boolean(), nullable=False),
    Column('conf', Numeric(10, 9), nullable=False)
)

metadata.create_all(engine)

class Owner(object):
    pass
class Cat(object):
    pass
class CatData(object):
    pass
class Device(object):
    pass
class CatLog(object):
    pass

mapper_registry.map_imperatively(Owner, owner)
mapper_registry.map_imperatively(Cat, cat)
mapper_registry.map_imperatively(CatData, cat_data)
mapper_registry.map_imperatively(Device, device)
mapper_registry.map_imperatively(CatLog, cat_log)

init_data()
