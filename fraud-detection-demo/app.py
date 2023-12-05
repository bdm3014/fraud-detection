import json
import prediction

# importing Flask libraries
from flask import Flask, request, Response, render_template, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint

# initializing Flask app
app = Flask(__name__)
CORS(app)

# defining the context for debugging
if __name__ == '__main__':
    app.run(debug=True)

#=== === === === === === === 

# defining error handler for error 404
@app.errorhandler(404)
def error_handler_404(error):
    return "Page not found", 404

# defining error handler for any other
@app.errorhandler(Exception)
def error_handler_exception(error):
    return "An error occurred", 500  # Internal Server Error

#=== === === === === === === 

# initializing Swagger module
SWAGGER_URL  = '/swagger'
SWAGGER_JSON = 'swagger.json'
SWAGGER_JSON_URL = '/'+SWAGGER_JSON
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    SWAGGER_JSON_URL,
    config={'app_name': "Fraud Prediction API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# defining Swagger endpoint
@app.route(SWAGGER_JSON_URL)
def swagger():
    with open(SWAGGER_JSON, 'r') as file:
        return jsonify(json.load(file))

#=== === === === === === ===

# defining root endpoint
@app.route('/')
def root():
    return index()  # redirecting to index page

# defining index page endpoint
@app.route('/index.html')
def index():
    return render_template('index.html')    # rendering from template

#=== === === === === === ===

# defining endpoint to locate customized field definition 
@app.route('/data/fields.json')
def get_data_fields():
    json_content = ""
    with open('data/fields.json', 'r') as file:
        json_content = file.read()
    response = Response(json_content, content_type='application/json')
    response.headers['Content-Disposition'] = 'filename=fields.json'
    return response     # opening the file and returning as web response

# defining endpoint to locate persisted data 
@app.route('/data/data.csv')
def get_data_file():
    data_content = ""
    with open('data/data.csv', 'r') as file:
        data_content = file.read()
    response = Response(data_content, content_type='text/txt')
    response.headers['Content-Disposition'] = 'filename=data.csv'
    return response     # opening the file and returning as web response

#=== === === === === === ===

# defining REST API endpoint for prediction
@app.route('/api/prediction', methods=['POST'])
def api_prediction_post():
    return prediction.post(request)     # calling the functionality from prediction library

#=== === === === === === ===