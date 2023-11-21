import camelot
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

class pdf2text:
    def __init__(self):
        """
        tables: list of parsed table string. Every element is a string for a table
        """
        self.tables = []
    def __call__(self, pdf_file_path: str):
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        print("Total tables extracted:", tables.n)
        for table in tables:
            self.tables.append(table.df.to_string().replace("\\n", ""))
        return self.tables
    def print_every_table(self, pdf_file_path: str):
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n\n".join(texts)
        text = text.replace("\\n", "")
        print(text)


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table)*norm(vector_from_keyword))


def main(keyword, pdf_file_path):    
    # parse pdf tables to string
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file_path)
    # convert string to vector
    txt2vec = text2vector()
    vectors = []
    for table in tables:
        vectors.append(txt2vec(table))
    # calculate cosine similarity
    cos_sim = cosine_sim()
    cosine_scores = []
    for vector in vectors:
        cosine_scores.append(cos_sim(vector, txt2vec(keyword)))
    # return table with highest similarity
    print(tables[np.argmax(cosine_scores)])

if __name__ == "__main__":
    # main("非監督式學習的應用", "docs/1.pdf")
    main("多細胞生物細胞膜的流動性較大，植物細胞膜的流動性較小。", "docs/2.pdf")
