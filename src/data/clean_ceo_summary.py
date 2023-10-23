# Cleans CEO summary Excel spreadsheet and converts to dictionary
import pandas as pd
import yaml
from utils.logs import get_logger


def import_ceo_summary(config_path, logger):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    ceo_summary = pd.read_excel(
        config["data_load"]["summary_filename"],
        sheet_name=config["data_load"]["summary_sheetname"],
        header=1,
    )
    ceo_summary = ceo_summary[
        ceo_summary[config["data_load"]["survey_id_column_name"]].notna()
    ]
    ceo_summary = ceo_summary.map(lambda x: x.lower() if isinstance(x, str) else x)
    ceo_summary_dict = ceo_summary.set_index(
        config["data_load"]["survey_id_column_name"]
    ).T.to_dict()
    for key, value in ceo_summary_dict.items():
        for ky, val in value.items():
            if isinstance(val, str):
                ceo_summary_dict[key][ky] = val.split(", ")
    logger.info("Summary data loaded")
    return ceo_summary_dict


def ceo_filter(ceo_dict, variable, string):
    filtered_dict = {}
    for key, value in ceo_dict.items():
        for ky, val in value.items():
            if (ky == variable) and (isinstance(val, list)):
                if string in val:
                    filtered_dict[key] = value
    return filtered_dict.keys()


def get_batch_numbers(batch_list):
    return [x[-3:].lower() for x in batch_list if len(x) > 2]


def id_ceo_batches(config_path, ceo_summary):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    ceo_batch_list = []
    for key, value in config["data_condition"]["train_filter_features"].items():
        ceo_batch_list.append(ceo_filter(ceo_summary, key, value))
    return get_batch_numbers(list(set.intersection(*map(set, ceo_batch_list))))
