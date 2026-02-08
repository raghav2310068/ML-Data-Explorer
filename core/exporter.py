class Exporter:
    def __init__(self, df):
        self.df = df

    def to_csv(self):
        return self.df.to_csv(index=False).encode("utf-8")
    
    def to_txt(self, text):
        return text.encode("utf-8")