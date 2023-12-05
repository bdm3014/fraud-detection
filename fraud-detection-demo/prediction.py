import json
import joblib
import numpy  as np
import pandas as pd

from flask import jsonify

#=== === === === === ===

# defining results as constants
RESULT           = "result"
RESULT_FRAUD     = "FRAUD"
RESULT_NOT_FRAUD = "NOT-FRAUD"
RESULT_UNKNOWN   = "UNKNOWN"
RESULT_ERROR     = "ERROR"

# defining main function to process POST method for prediction API 
def post(request):
    homo_json = ""
    try:
        # getting the JSON data from the request
        data_json = request.json
        
        # getting field features
        names_list, fields_dict = load_fields()
        
        # validating the presence of parameters
        validate(names_list, fields_dict, data_json)
        
        # homologating values
        homo_json = homologate(names_list, fields_dict, data_json)
        
        # getting the unified prediction
        result = unified_prediction(homo_json)
        
        # saving the data and result
        save_data(names_list, data_json, homo_json, result)
        
        # returning the OK result as a JSON
        response = {RESULT: result}
        return jsonify(response)

    except Exception as ex:
        # saving the data and result
        error_msg = RESULT_ERROR+': '+str(ex.args[0])
        save_data(names_list, data_json, homo_json, error_msg)

        # returning the ERROR result as a JSON and 400 response
        response  = {RESULT: error_msg}
        return jsonify(response), 400  # 400 Bad Request

#=== === === === === ===
#=== === === === === ===

# loading the field definition file 
# returning a list with all field names, and the field definition as a dictionary 
def load_fields():
    # loading the field definition file
    fields_path = "data/fields.json"
    with open(fields_path, "r") as json_file:
        fields_list = json.load(json_file)
    
    # getting a list with all field names
    names_list = [field["name"] for field in fields_list]
    fields_dict = { }
    for name in names_list:
        fields_dict[name] = get_field_by_name(fields_list, name)

    # returning both, list with all field names, and the field definition
    return names_list, fields_dict

# getting the field definition from a dictionary using their name
# returning the field entry (with their properties) or None
def get_field_by_name(fields_list, name):
    for field in fields_list:
        if field.get("name") == name:
            return field
    return None

#=== === === === === ===

# validating fields and data in the JSON received
def validate(names_list, fields_dict, data_json):
    # validating fields that are not expected
    params_unknown_list = [ ]
    for key, value in data_json.items():
        if not key in names_list:
            params_unknown_list.append(key)
    if len(params_unknown_list) > 0:
        raise Exception("Unknown Parameters: "+", ".join(params_unknown_list))
    
    # validating fields that are required
    params_missed_list = [ ]
    for name in names_list:
        if fields_dict.get(name)["required"] in (True, "true"):
            if not name in data_json:
                params_missed_list.append(name)
            elif is_null_or_empty(data_json[name]):
               params_missed_list.append(name)
    if len(params_missed_list) > 0:
        raise Exception("Required Parameters: "+", ".join(params_missed_list))
    
    # validating numeric data to be in expected ranges
    params_nums_out_range_list = [ ]
    for name in names_list:
        if name in data_json and not is_null_or_empty(data_json[name]) \
           and fields_dict.get(name)["type"] in ("number", "decimal"):
                if "min" in fields_dict.get(name):
                    if float(data_json[name]) < int(fields_dict.get(name)["min"]):
                        params_nums_out_range_list.append(name)
                if "max" in fields_dict.get(name):
                    if float(data_json[name]) > int(fields_dict.get(name)["max"]):
                        params_nums_out_range_list.append(name)
    if len(params_nums_out_range_list) > 0:
        raise Exception("Out of Range Parameters: "+", ".join(params_nums_out_range_list))

# identifying if one object is null or empty
def is_null_or_empty(obj):
    if obj is None:
        return True
    elif obj is np.nan:
        return True
    elif isinstance(obj, str) and not obj.strip():
        return True
    elif isinstance(obj, (list, tuple, dict, set)) and not obj:
        return True
    else:
        return False

#=== === === === === ===

# homologating and getting the received data into the form the models need
def homologate(names_list, fields_dict, data_json):
    homo_json = { } # recipient of homologated data

    # looping over all defined fields ... 
    for name in names_list:
        # getting the final name to send to the model
        homo_name = name
        if 'encoded_name' in fields_dict[name]: 
            homo_name = fields_dict[name]['encoded_name']
        
        # if the field is not received, setting null and skipping this field
        if not name in data_json:
            homo_json[name]      = np.nan
            homo_json[homo_name] = np.nan
            continue
        
        # if the value is null or empty, setting null and skipping this field
        value = data_json[name]
        if is_null_or_empty(value):
            homo_json[name]      = np.nan
            homo_json[homo_name] = np.nan
            continue
        
        # setting the initial fields with the original value received
        homo_json[name]      = value
        homo_json[homo_name] = value              
        
        # if there is defined encoder function ...
        try:
            if 'encoded_func' in fields_dict[name] and fields_dict[name]['encoded_func']:
                # getting the function, executing, and getting the result
                variables = { 'value': value }
                exec(fields_dict[name]['encoded_func'], globals(), variables)
                result = variables.get('result')
                # setting the result as homogenized field
                homo_json[homo_name] = result
        except:
            raise Exception("Problem executing encoded_func for: "+name)
        
        # if there is not defined encoded list and value is number ...
        if not 'encoded_list' in fields_dict[name]:
            # if there is the field and is numeric, setting values as float and skipping this field
            if name      in homo_json and is_number(value):                homo_json[name]      = float(value)
            if homo_name in homo_json and is_number(homo_json[homo_name]): homo_json[homo_name] = float(homo_json[homo_name])
            continue            
        
        # getting the encoded list dictionary
        values_dict = { }
        list_filename = "models/"+fields_dict[name]['encoded_list']
        try:
            with open(list_filename, "r") as json_file:
                values_dict = json.load(json_file)
        except:
            raise Exception("Problem opening the file: "+list_filename)
        
        # if value is found, set the value as float and continue
        if value in values_dict:
            homo_json[homo_name] = float(values_dict[value])
            continue
        elif str(value)+".0" in values_dict:
            homo_json[homo_name] = float(values_dict[str(value)+".0"])
            continue
        
        # reached here, nothing more to do, setting null
        homo_json[homo_name] = np.nan
        
    # processing special case that cannot manage in other way: addr -> addr_target_encoded
    try:
        if 'addr1' in names_list and 'addr2' in names_list:
            value = str(data_json['addr1'])+".0_"+str(data_json['addr2'])+".0"
            if value == 'nan_nan': value = np.nan
            values_dict = { }
            list_filename = "models/list-addr-to-addr_target_encoded.json"
            with open(list_filename, "r") as json_file:
                values_dict = json.load(json_file)
            name      = "addr"
            homo_name = "addr_target_encoded"
            homo_json[name]      = value
            homo_json[homo_name] = float(values_dict[value]) if value in values_dict else float(values_dict['other_other'])
    except:
        raise Exception("Problem processing data: addr")
    
    # processing special case that cannot manage in other way: P_emaildomain_suffix_us
    try:
        if 'P_emaildomain' in names_list:
            value = str(data_json['P_emaildomain']).lower()
            homo_name = "P_emaildomain_suffix_us"
            us_emails = ['gmail', 'net', 'edu']
            valid_suffix = lambda email, valid_suffixes: any(email.endswith(suffix) for suffix in valid_suffixes)
            homo_json[homo_name] = 1 if valid_suffix(value, us_emails) else 0
    except:
        raise Exception("Problem processing data: P_emaildomain_suffix_us")

    ## showing both original and homogenized data for debugging purposes
    #print("data_json = ", data_json)
    #print("homo_json = ", homo_json)

    # returning the homologated data
    return homo_json

# defining one input object is a number or not
def is_number(input):
    try:
        float(input)
        return True
    except ValueError:
        return False

#=== === === === === ===

def unified_prediction(homo_json):
    # getting factors of both predictions
    result01, result01_proba = prediction(homo_json, 'model01_xgboost_joblib')
    result02, result02_proba = prediction(homo_json, 'model02_lightgbm_joblib')

    # choosing final prediction (abandoned criteria)
    print("XGBoost ", result01, ",", result01_proba)
    print("LightGBM", result02, ",", result02_proba)
    # result = result01 if result01         ==  result02 else \
    #      RESULT_FRAUD if RESULT_FRAUD     in (result01,result02) else \
    #  RESULT_NOT_FRAUD if RESULT_NOT_FRAUD in (result01,result02) else \
    #    RESULT_UNKNOWN if RESULT_UNKNOWN   in (result01,result02) else \
    #      RESULT_ERROR
    
    # choosing final prediction
    # 1. Run XGB, get the probas. If the proba >0.5 assign 1
    # 2. If xgb proba <=0.5 then run lgbm
    # 3. If lgbm proba >0.5 assign 1, otherwise 0
    result = RESULT_UNKNOWN
    if   result01_proba > 0.5: result = RESULT_FRAUD
    elif result02_proba > 0.5: result = RESULT_FRAUD
    else: result = RESULT_NOT_FRAUD

    # returning the chosen prediction
    return result

# getting the prediction using the homologated data and the model name
def prediction(homo_json, model_name):
    # loading the model from file
    model_filename = 'models/'+model_name+'.pkl'
    model_loaded = joblib.load(model_filename)

    # identifing the classifier type
    from xgboost.sklearn  import XGBClassifier
    from lightgbm.sklearn import LGBMClassifier
    isXGBClassifier  = isinstance(model_loaded, XGBClassifier)
    isLGBMClassifier = isinstance(model_loaded, LGBMClassifier)

    # creating the feature/input dataframe
    homo_json["TransactionID"] = 0
    dataframe_X = pd.DataFrame(homo_json, index=[0])
    dataframe_X.set_index("TransactionID", inplace=True)
    # restructuring the dataframe in the shape (elements and order) the model needs
    if isXGBClassifier:  dataframe_X = dataframe_X[model_loaded.get_booster().feature_names]
    if isLGBMClassifier: dataframe_X = dataframe_X[model_loaded.feature_name_]

    # calling the model's predict method
    y_pred = model_loaded.predict(dataframe_X)
    # calling the model's predict probability method
    y_pred_proba = model_loaded.predict_proba(dataframe_X)[:, 1]  

    # getting the result, interpreting, and returning the prediction
    result = str(y_pred[0])
    return RESULT_FRAUD if result == "1" else RESULT_NOT_FRAUD if result == "0" else RESULT_UNKNOWN, y_pred_proba

#=== === === === === ===

# saving the input data and the result into a file
def save_data(names_list, data_json, homo_json, result):
    data_filename = "data/data.csv"
    with open(data_filename, 'a') as file:
        result_normalized = '' if is_null_or_empty(result) else result.replace(',',';')
        homo_strg = str(homo_json).replace(',','; ').replace('\'','').replace('\"','')
        new_line = result_normalized + ', ' + json_csv_line(names_list, data_json) + ', ' + homo_strg
        file.write('\n'+new_line)

# creating a txt line to save in a csv fdile from a JSON data 
def json_csv_line(names_list, data_json):
    DATA_EMPTY_VALUE = "-"
    values_list = [ ]
    for name in names_list:
        if name in data_json:
            if is_null_or_empty(data_json[name]):
                values_list.append(DATA_EMPTY_VALUE)
            else: values_list.append(str(data_json[name]).replace(',',';'))
        else: values_list.append(DATA_EMPTY_VALUE)
    return ', '.join(values_list)

#=== === === === === ===