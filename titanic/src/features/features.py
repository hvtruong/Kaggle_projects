import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for training:
    - drop Name, Ticket, Fare, and Cabin
    - fill missing cells of Age and Embarke
    - combine SibSp and Parch into 1 column FamilySize
    - split age to 5 groups: Infant, child, Teenager, Adult, and Senior (0-4)
    - encode categorical columns: Sex, Embarked, PClass
    - add 2 significant indicators: is_infant_or_female, is_family_size_2_to_4_or_upper_class
    """
    df = df.drop(['Name', 'Ticket', 'Fare', 'Cabin'], axis=1)
    
    # Fill missing ages with the median of people with the same PClass and Sex
    df = df.fillna({'Age': df.groupby(['Pclass', 'Sex'])['Age'].transform('median')})
    # Fill missing Embarked with the most frequent port of people with the same PClass and Sex
    df = df.fillna({'Embarked': df.groupby(['Pclass', 'Sex'])['Embarked'].transform(lambda x: x.mode()[0])})
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df = df.drop(['SibSp', 'Parch'], axis=1)
    
    df['Age'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 65, 150], labels=[0, 1, 2, 3, 4])
    
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'S': 1, 'Q': 2})
    df['Pclass'] = df['Pclass'] - 1
    
    df['is_infant_or_female'] = ((df['Age'] == 0) | (df['Sex'] == 1)).astype('int')
    df['is_family_size_2_to_4_or_upper_class'] = ((df['FamilySize'] >= 2) & (df['FamilySize'] <= 4) | (df['Pclass'] == 2)).astype('int')

    return df