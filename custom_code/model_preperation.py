from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import label_binarize
import pandas as pd

def get_title(names):
    titles = names.str.extract(r' ([A-Za-z]+)\.', expand=False)
    titles = titles.replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
    titles = titles.replace('Mlle','Miss')
    titles = titles.replace('Ms','Miss')
    titles = titles.replace('Mme','Mrs')
    return titles

class ModelPreperation(TransformerMixin):
    #Class Constructor 
    def __init__( self ):
        self.title_encoder = LabelEncoder()
        pass
     
    def fit( self, X, y=None):
        self.default_age = X.Age.median()
        self.default_price = X.Fare.mean()
        self.default_harbor = X.Embarked.mode()[0]
        self.embarked_categories = list(X.Embarked.dropna().unique())
        self.title_encoder.fit(get_title(X.Name))
        return self
        
    def transform(self, df):
        df = df.copy()
        df["Age"].fillna(self.default_age, inplace=True)
        df["Fare"].fillna(self.default_price, inplace=True)
        df["Embarked"].fillna(self.default_harbor, inplace=True)
        df['Sex'] = label_binarize(df.Sex, ['male', 'female'])
        df["has_cabin"] = ~df.Cabin.isna()
        df["title"] = self.title_encoder.transform(get_title(df.Name))
        df = df.drop(columns=["Cabin","Name","Ticket","PassengerId"], errors='ignore')
        df["Embarked"] = df.Embarked.astype("category").cat.set_categories(self.embarked_categories)
        df = pd.get_dummies(df, ["Embarked"], columns=["Embarked"], drop_first=True)
        return df
