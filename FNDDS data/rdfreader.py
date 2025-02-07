import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

from rdflib import Graph

# ����һ�� RDF ͼ
g = Graph()
df = pd.read_excel("/2021-2023 FNDDS At A Glance - Portions and Weights.xlsx")


g.parse("/2021-2023 FNDDS At A Glance - Portions and Weights.xlsx", format="xml")  # �����ļ���ʽѡ�������ʽ


EX = Namespace("http://example.org/food#")
g.bind("ex", EX)

for _, row in df.iterrows():
    #�� Food code ��Ϊ URI
    food_uri = URIRef(EX[str(row['Food code'])])
    g.add((food_uri, RDF.type, EX.Food))
    g.add((food_uri, EX.mainFoodDescription, Literal(row['Main food description'], datatype=XSD.string)))
    g.add((food_uri, EX.wweiaCategoryNumber, Literal(row['WWEIA Category number'], datatype=XSD.integer)))
    g.add((food_uri, EX.wweiaCategoryDescription, Literal(row['WWEIA Category description'], datatype=XSD.string)))
    g.add((food_uri, EX.portionDescription, Literal(row['Portion description'], datatype=XSD.string)))
    g.add((food_uri, EX.portionWeight, Literal(row['Portion weight (g)'], datatype=XSD.float)))

# ����Ϊ RDF ��ʽ�ļ������� Turtle ��ʽ��
output_path = "food_data.rdf"
g.serialize(destination=output_path, format="turtle")

print(f"RDF data has been saved to {output_path}")



