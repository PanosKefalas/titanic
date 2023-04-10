from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class titanicEncoder(BaseEstimator, TransformerMixin):

    def _init_(self):
        self

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        encodeEmbarkation = pd.get_dummies(X['Embarked'])
        embarkationEncoded = pd.concat([X, encodeEmbarkation], axis=1)
        dfEncoded = embarkationEncoded.drop(columns='Embarked')

        dfEncoded['Sex'] = dfEncoded['Sex'].replace(['female', 'male'], [0,1])
        dfEncoded['Cabin'] = dfEncoded['Cabin'].notna().astype('int')
        
        return dfEncoded