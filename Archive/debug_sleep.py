import pandas as pd
df = pd.read_excel('Data compiler/survey_1_processed.xlsx')
col = "Ad'n: How would you rate your sleep quality last night? "
print(df[col].unique())
