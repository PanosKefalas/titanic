from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd

class titanicEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, dropUnusedColumns=True):
        self.dropUnusedColumns = dropUnusedColumns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        encodeEmbarkation = pd.get_dummies(X['Embarked'])
        embarkationEncoded = pd.concat([X, encodeEmbarkation], axis=1)
        dfEncoded = embarkationEncoded.drop(columns='Embarked')

        dfEncoded['Sex'] = dfEncoded['Sex'].replace(['female', 'male'], [1,0])
        dfEncoded['Cabin'] = dfEncoded['Cabin'].notna().astype('int')
        
        if self.dropUnusedColumns:
            dfEncoded = dfEncoded.drop(columns=['PassengerId', 'Name', 'Ticket'])

        return dfEncoded
    
class imputeColumnMean(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['Age']):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        dfFilled = X.copy()
        dfFilled[self.columns] = X[self.columns].fillna(X[self.columns].mean())
        return dfFilled
    

def transformTitanicDf(X, y=None):
    encoder = titanicEncoder(dropUnusedColumns=True)
    dfEncoded = encoder.fit_transform(X)

    cleanPipeline = Pipeline([
    ('imputer', imputeColumnMean()),
    ('scaler', MinMaxScaler())
    ])

    dfTransformed = dfEncoded.copy()
    dfTransformed[dfTransformed.columns] = cleanPipeline.fit_transform(dfEncoded)

    dfTransformed = dfTransformed.reset_index(drop=True)

    return dfTransformed