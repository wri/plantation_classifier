# CLeans CEO summary Excel spreadsheet and converts to dictionary

import pandas as pd

ceo_filepath = '../data/Training_Data.xlsx'
survey_sheetname = 'CEO_Surveys'
ceo_survey_id_col = 'CEO Survey'

def import_ceo_summary(ceo_filepath, survey_sheetname):
    ceo_summary = pd.read_excel(ceo_filepath, sheet_name=survey_sheetname, header=1)
    ceo_summary = ceo_summary[ceo_summary[ceo_survey_id_col].notna()]
    ceo_summary = ceo_summary.map(lambda x: x.lower() if isinstance(x, str) else x)
    ceo_summary_dict = ceo_summary.set_index(ceo_survey_id_col).T.to_dict()
    return ceo_summary_dict

