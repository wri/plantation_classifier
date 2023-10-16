# CLeans CEO summary Excel spreadsheet and converts to dictionary
import pandas as pd
from typing import Text
import yaml
from utils.logs import get_logger

def import_ceo_summary(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    logger = get_logger('CEO_SUMMARY_LOAD', log_level=config['base']['log_level'])
    ceo_summary = pd.read_excel(config['data_load']['summary_filename'], sheet_name=config['data_load']['summary_sheetname'], header=1)
    ceo_summary = ceo_summary[ceo_summary[config['data_load']['survey_id_column_name']].notna()]
    ceo_summary = ceo_summary.map(lambda x: x.lower() if isinstance(x, str) else x)
    ceo_summary_dict = ceo_summary.set_index(config['data_load']['survey_id_column_name']).T.to_dict()
    for key, value in ceo_summary_dict.items():
        for ky, val in value.items():
            if isinstance(val, str):
                ceo_summary_dict[key][ky] = val.split(', ')
    logger.info('Summary data loaded')
    return ceo_summary_dict