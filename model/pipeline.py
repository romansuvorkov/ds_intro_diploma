from datetime import time

import datetime
import dill
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def merge_data(sessions, hits):
    import pandas as pd
    sessions_df = sessions.copy()
    hits_df = hits.copy()
    df_hits_filtered = hits_df[((hits_df['event_action'] == 'sub_car_claim_click') | (
            hits_df['event_action'] == 'sub_car_claim_submit_click') | (
                                        hits_df['event_action'] == 'sub_open_dialog_click') | (
                                        hits_df['event_action'] == 'sub_custom_question_submit_click') | (
                                        hits_df['event_action'] == 'sub_call_number_click') | (
                                        hits_df['event_action'] == 'sub_callback_submit_click') | (
                                        hits_df['event_action'] == 'sub_submit_success') | (
                                        hits_df['event_action'] == 'sub_car_request_submit_click'))]
    hits_df = df_hits_filtered.groupby('session_id').count()
    merged_df = pd.merge(sessions_df, hits_df[['event_action']], on='session_id', how='left')
    merged_df.loc[merged_df['event_action'].isna(), 'event_action'] = 0
    merged_df.loc[merged_df['event_action'] > 0, 'event_action'] = 1
    return merged_df


def clean_df(sessions):
    def get_boundaries(data):
        q25 = data.quantile(0.25)
        q75 = data.quantile(0.75)
        iqr = q75 - q25
        boundaries = (round(q25 - 1.5 * iqr), round(q75 + 1.5 * iqr))
        return boundaries

    sessions_df = sessions.copy()
    space_mask = ((sessions_df['device_brand'] == '') | (sessions_df['device_brand'].str.isspace()))
    sessions_df.loc[space_mask, 'device_brand'] = None
    not_set_brand_mask = (sessions_df['device_brand'] == '(not set)')
    sessions_df.loc[not_set_brand_mask, 'device_brand'] = None
    not_set_os_mask = (sessions_df['device_os'] == '(not set)')
    sessions_df.loc[not_set_os_mask, 'device_os'] = None
    sessions_df.loc[not_set_os_mask, 'device_browser'] = None
    df_apple_sessions_mobile = ((sessions_df['device_brand'] == 'Apple') & (sessions_df['device_os'].isna()) & (
            sessions_df['device_category'] == 'mobile'))
    df_apple_sessions_tablet = ((sessions_df['device_brand'] == 'Apple') & (sessions_df['device_os'].isna()) & (
            sessions_df['device_category'] == 'tablet'))
    df_apple_sessions_desktop = ((sessions_df['device_brand'] == 'Apple') & (sessions_df['device_os'].isna()) & (
            sessions_df['device_category'] == 'desktop'))
    sessions_df.loc[df_apple_sessions_mobile, 'device_os'] = 'iOS'
    sessions_df.loc[df_apple_sessions_tablet, 'device_os'] = 'iOS'
    sessions_df.loc[df_apple_sessions_desktop, 'device_os'] = 'Macintosh'
    df_safari_sessions_mobile_os_na = (
            (sessions_df['device_browser'].str.contains('Safari')) & (sessions_df['device_os'].isna()) & (
            sessions_df['device_category'] == 'mobile'))
    df_safari_sessions_tablet_os_na = (
            (sessions_df['device_browser'].str.contains('Safari')) & (sessions_df['device_os'].isna()) & (
            sessions_df['device_category'] == 'tablet'))
    df_safari_sessions_desktop_os_na = (
            (sessions_df['device_browser'].str.contains('Safari')) & (sessions_df['device_os'].isna()) & (
            sessions_df['device_category'] == 'desktop'))
    sessions_df.loc[df_safari_sessions_mobile_os_na, 'device_os'] = 'iOS'
    sessions_df.loc[df_safari_sessions_tablet_os_na, 'device_os'] = 'iOS'
    sessions_df.loc[df_safari_sessions_desktop_os_na, 'device_os'] = 'Macintosh'
    df_safari_sessions_brand_na = (
            sessions_df['device_browser'].str.contains('Safari') & (sessions_df['device_brand'].isna()))
    apple_mac_mask = ((sessions_df['device_os'] == 'Macintosh') & (sessions_df['device_brand'].isna()))
    sessions_df.loc[df_safari_sessions_brand_na, 'device_brand'] = 'Apple'
    sessions_df.loc[apple_mac_mask, 'device_brand'] = 'Apple'
    sessions_df.loc[sessions_df['device_brand'].isna(), 'device_brand'] = 'Unknown'
    boundaries = get_boundaries(sessions_df['visit_number'])
    is_outlier = (sessions_df['visit_number'] < boundaries[0]) | (sessions_df['visit_number'] > boundaries[1])
    sessions_df.loc[is_outlier, 'visit_number'] = boundaries[1]
    boundaries = None
    sessions_df['visit_time'] = sessions_df['visit_time'].astype('str')
    sessions_df['event_action'] = sessions_df['event_action'].astype('int64')
    sessions_df = sessions_df[sessions_df['utm_source'].notna()]
    sessions_df = sessions_df[sessions_df['device_browser'].notna()]
    sessions_df = sessions_df[sessions_df['utm_campaign'].notna()]
    sessions_df = sessions_df[sessions_df['utm_adcontent'].notna()]
    return sessions_df


def decode_location(dataframe):
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    import os.path
    geolocator = Nominatim(user_agent='my_application')
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    def encode_city_location(city_name):
        location = geocode(city_name, timeout=None)
        if not location:
            location = geocode('(not set)')
        return [location.latitude, location.longitude]

    def get_city_names_list(df_column):
        output = df_column.unique()
        return output

    def calculate_city_coords(cities_list):
        import json
        path = './model/data/city_coords.json'
        check_file = os.path.isfile(path)
        output_update = False
        if check_file:
            with open(path, 'r') as openfile:
                output = json.load(openfile)
        else:
            output_update = True
            output = dict()
        for city_name in cities_list:
            if city_name not in output:
                output[city_name] = encode_city_location(city_name)
        if output_update:
            with open(path, 'w') as outfile:
                json.dump(output, outfile)
        return output

    df_output = dataframe.copy()
    city_names_list = get_city_names_list(dataframe['geo_city'])
    city_coords_list = calculate_city_coords(city_names_list)
    for item in city_coords_list:
        df_output.loc[df_output['geo_city'] == item, ['city_lat', 'city_long']] = city_coords_list[item][0], \
            city_coords_list[item][1],
    df_output = df_output.drop([
        'geo_city',
    ], axis=1)
    return df_output


def decode_timestamps(dataframe):
    import pandas as pd

    def calculate_hour(income_string):
        splitted_string = income_string.split(':')
        hour = int(splitted_string[0])
        minutes = int(splitted_string[1])
        if minutes > 30:
            if hour == 23:
                hour = 0
            else:
                hour += 1
        return hour
    df_output = dataframe.copy()
    df_output['visit_date'] = pd.to_datetime(df_output['visit_date'])
    df_output['visit_dayofweek'] = df_output['visit_date'].dt.dayofweek
    df_output['visit_day'] = df_output['visit_date'].dt.day
    df_output['visit_month'] = df_output['visit_date'].dt.month
    df_output['visit_hour'] = df_output['visit_time'].apply(lambda x: calculate_hour(x))
    df_output = df_output.drop([
        'visit_date',
        'visit_time',
    ], axis=1)
    return df_output


def decode_device_sizes(dataframe):
    import pandas as pd

    def encode_pixels(income_row):
        income_string = income_row['device_screen_resolution']
        if income_string == '(not set)':
            return [10, 10]
        splitted_string = income_string.split('x')
        return [int(splitted_string[0]), int(splitted_string[1])]

    df_output = dataframe.copy()
    pixel_numbers = df_output.apply(lambda x: encode_pixels(x), axis=1, result_type='expand')
    pixel_numbers.columns = ['pixel_width', 'pixel_height']
    df_output = pd.concat([df_output.reset_index(drop=True), pixel_numbers.reset_index(drop=True)], axis=1)
    df_output = df_output.drop([
        'device_screen_resolution',
    ], axis=1)
    df_output['pixel_width'] = df_output['pixel_width'].apply(lambda x: x / 10)
    df_output['pixel_width'] = df_output['pixel_width'].round(0)
    df_output['pixel_height'] = df_output['pixel_height'].apply(lambda x: x / 10)
    df_output['pixel_height'] = df_output['pixel_height'].round(0)
    return df_output


def create_short_browser(dataframe):
    df_output = dataframe.copy()
    df_output['short_browser'] = df_output['device_browser'].apply(lambda x: x.split(' ')[0])
    df_output = df_output.drop([
        'device_browser',
    ], axis=1)
    return df_output


def downsampling_dataset(input_df):
    import pandas as pd
    initial_df = input_df.copy()
    df_majority = initial_df[initial_df['event_action'] == 0]
    df_minority = initial_df[initial_df['event_action'] == 1]
    df_minority_len = df_minority.shape[0]
    df_majority_downsampled = resample(df_majority,
                                       replace=False,
                                       n_samples=df_minority_len,
                                       random_state=852)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled


def drop_unnecessary_columns(df):
    output = df.copy()
    output = output.drop([
        'device_model',
        'utm_keyword',
        'device_os',
        'session_id',
        'client_id',
        'geo_country',
    ], axis=1)
    return output


def main():
    print('Event Prediction Pipeline')
    df_sessions = pd.read_pickle('./model/data/ga_sessions.pkl')
    df_hits = pd.read_pickle('./model/data/ga_hits.pkl')
    merged_df = merge_data(df_sessions, df_hits)
    merged_df = downsampling_dataset(merged_df)
    merged_df = clean_df(merged_df)
    X = merged_df.drop(['event_action'], axis=1)
    y = merged_df['event_action']
    df_cleaner = Pipeline(steps=[
        ('drop_columns', FunctionTransformer(drop_unnecessary_columns)),
    ])

    feature_creation = Pipeline(steps=[
        ('create_short_browser', FunctionTransformer(create_short_browser)),
        ('decode_device_size', FunctionTransformer(decode_device_sizes)),
        ('decode_timestamp', FunctionTransformer(decode_timestamps)),
        ('decode_geo_location', FunctionTransformer(decode_location)),
    ])

    standart_scaling = Pipeline(steps=[
        ('std_scaler', StandardScaler()),
    ])

    min_max_scaling = Pipeline(steps=[
        ('mm_scaler', MinMaxScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ])

    column_transformer = ColumnTransformer(transformers=[
        (
            'numerical_std', standart_scaling,
            ['visit_number', 'visit_dayofweek', 'visit_day', 'visit_month', 'visit_hour',
             'pixel_width', 'pixel_height']),
        ('numerical_mm', min_max_scaling, ['city_lat', 'city_long']),
        ('categorical', categorical_transformer, [
            'utm_source',
            'utm_medium',
            'utm_campaign',
            'utm_adcontent',
            'device_category',
            'device_brand',
            'short_browser',
        ])
    ])

    preprocessor = Pipeline(steps=[
        ('filter', df_cleaner),
        ('feature_creator', feature_creation),
        ('column_transformer', column_transformer),
    ])

    models = (
        MLPClassifier(random_state=42, max_iter=19000, hidden_layer_sizes=5, learning_rate='adaptive', solver='lbfgs'),
        RandomForestClassifier(random_state=42, bootstrap=True, n_estimators=200, min_samples_split=15),
        SVC(),
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='roc_auc')
        print(f'model: {type(model).__name__}, roc_auc_mean: {score.mean():.2f}, roc_auc_std: {score.std():.2f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, roc_auc: {best_score:.2f}')
    with open('model/best_pipe.pkl', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Sberauto event prediction pipeline',
                'author': 'Roman Suvorkov',
                'version': 1,
                'date': datetime.datetime.now(),
                'type': type(best_pipe.named_steps['classifier']).__name__,
                'roc_auc': best_score,
            }
        }, file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
