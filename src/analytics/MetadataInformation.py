import pandas as pd

N_SECONDS_PER_DAY = 86400
GRAMS_PER_HOUR = 225
N_HOURS_PER_DAY = 24
GRAMS_PER_KG = 1000
TREES_PER_KG_OF_CO2 = 0.012
GB_ELECTRICITY_TRAINING = 5.7
GB_ELECTRICITY_TESTING = 2
GB_GAS_TRAINING = 4.8
GB_GAS_TESTING = 1.6


class MetadataInformation:

    @staticmethod
    def get_number_of_meterIDs(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
        elif dataset == "Solar (Ausgrid)":
            dataset = "solar"
            return 0
        return len(pd.read_csv('./script_results/' + dataset + '_detector_comparer_results.csv')['meterID'].unique())

    @staticmethod
    def get_number_of_training_weeks(dataset):
        N_OBS_PER_WEEK = 336

        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
            return int(len(pd.read_csv('./script_results/' + dataset + '_training_data/1540_0_60.csv')['DT'].unique()) / N_OBS_PER_WEEK)
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
            return int(len(pd.read_csv('./script_results/' + dataset + '_training_data/1838_0_60.csv')['DT'].unique()) / N_OBS_PER_WEEK)
        elif dataset == "Solar (Ausgrid)":
            dataset = "solar"
            return int(len(pd.read_csv('./script_results/' + dataset + '_training_data/1_0_50.csv')['DT'].unique()) / N_OBS_PER_WEEK)
        return 0

    @staticmethod
    def get_number_of_testing_weeks(dataset):
        N_OBS_PER_WEEK = 336

        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
            return int(len(pd.read_csv('./script_results/' + dataset + '_testing_data/1540_61_75.csv')['DT'].unique()) / N_OBS_PER_WEEK)
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
            return int(len(pd.read_csv('./script_results/' + dataset + '_testing_data/1838_61_77.csv')['DT'].unique()) / N_OBS_PER_WEEK)
        elif dataset == "Solar (Ausgrid)":
            dataset = "solar"
            return int(len(pd.read_csv('./script_results/' + dataset + '_testing_data/1_51_101.csv')['DT'].unique()) / N_OBS_PER_WEEK)
        return 0

    @staticmethod
    def get_number_of_normal_scenarios(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
        elif dataset == "Solar (Ausgrid)":
            dataset = "solar"
            return 0
        list_of_attacks = pd.read_csv('./script_results/' + dataset + '_detector_comparer_results.csv')['attack'].unique().tolist()
        return list_of_attacks.count('False')

    @staticmethod
    def get_number_of_attack_scenarios(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
        elif dataset == "Solar (Ausgrid)":
            dataset = "solar"
            return 0

        list_of_attacks = pd.read_csv('./script_results/' + dataset + '_detector_comparer_results.csv')['attack'].unique().tolist()
        return len(list_of_attacks) - list_of_attacks.count('False')

    @staticmethod
    def get_number_of_detectors(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
        elif dataset == "Solar (Ausgrid)":
            return 0

        return len(pd.read_csv('./script_results/' + dataset + '_detector_comparer_results.csv')['detector'].unique().tolist())

    @staticmethod
    def get_experimentation_time(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            dataset = "electricity"
        elif dataset == "Gas (ISSDA-CER)":
            dataset = "gas"
        elif dataset == "Solar (Ausgrid)":
            dataset = "solar"
            return 0

        df = pd.read_csv('./script_results/' + dataset + '_detector_comparer_results.csv')

        time_model_prediction = df['time_model_prediction'].sum()
        time_model_creation = df['time_model_creation'].drop_duplicates().sum()

        return round((time_model_creation + time_model_prediction) / N_SECONDS_PER_DAY, 2)

    @staticmethod
    def get_carbon_footprint(dataset):
        # 52 or 234 g/CO2 per hour (https://www.ecoembes.com/es/planeta-recicla/blog/los-ordenadores-tambien-emiten-co2)
        # We consider 225 g/CO2, as we do intensive work
        return round(GRAMS_PER_HOUR * N_HOURS_PER_DAY * MetadataInformation.get_experimentation_time(dataset) / GRAMS_PER_KG, 2)

    @staticmethod
    def get_number_of_trees(dataset):
        # https://rumboeconomico.net/cuantos-arboles-debemos-plantar-para-compensar-el-co2-que-producimos/
        # 6 ton/CO2 per year per human
        # 6 trees per month to counter a human carbon footprint
        # So we need 6*12 = 72 trees to counter 6 ton/CO2
        # We'll need to plant 0.012 trees per kg of CO2
        return round(MetadataInformation.get_carbon_footprint(dataset) * TREES_PER_KG_OF_CO2, 1)

    @staticmethod
    def get_gb_of_training_dataset(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            return GB_ELECTRICITY_TRAINING
        elif dataset == "Gas (ISSDA-CER)":
            return GB_GAS_TRAINING
        elif dataset == "Solar (Ausgrid)":
            return 0

    @staticmethod
    def get_gb_of_testing_dataset(dataset):
        if dataset == "Electricity (ISSDA-CER)":
            return GB_ELECTRICITY_TESTING
        elif dataset == "Gas (ISSDA-CER)":
            return GB_GAS_TESTING
        elif dataset == "Solar (Ausgrid)":
            return 0
