import numpy as np
import joblib
import pickle
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)
model = joblib.load("rt_lm.pkl")



@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')

form_values=[]
@app.route('/QuestionOne', methods=["GET", "POST"])
def QuestionOne():
    form_values.append(list(request.form.values()))
    return render_template('QuestionOne.html', 
    gender_=[{'gender': 'Male'}, {'gender': 'Female'}])

@app.route('/QuestionTwo', methods=["GET", "POST"])
def QuestionTwo():
    form_values.append(list(request.form.values()))
    return render_template('QuestionTwo.html', 
    any_children_=[{'any_children': 'Yes'}, {'any_children': 'No'}])
    
@app.route('/QuestionThree', methods=["GET", "POST"])
def QuestionThree():
    form_values.append(list(request.form.values()))
    return render_template('QuestionThree.html', 
    years_married_=[{'years_married': '3 months or less'}, {'years_married': '4-6 months'},{'years_married': '6 months'},
        {'years_married': '1-2 years'}, {'years_married': '3-5 years'}, {'years_married': '6-8 years'}, {'years_married': '9-11 years'}, {'years_married': '12 or more years'}])

@app.route('/QuestionFour', methods=["GET", "POST"])
def QuestionFour():
    form_values.append(list(request.form.values()))
    return render_template('QuestionFour.html', 
    age_=[{'age': 'under 20'}, {'age': '20-24'}, {'age': '25-29'}, {'age': '30-34'}, {'age': '35-39'}, {'age': '40-44'},
              {'age': '45-49'}, {'age': '50-54'}, {'age': '55 and over'}])

@app.route('/QuestionFive', methods=["GET", "POST"])
def QuestionFive():
    form_values.append(list(request.form.values()))
    return render_template('QuestionFive.html', 
    religiousness_=[{"religiousness":"anti"}, {"religiousness":"not at all"}, {"religiousness":"slightly"}, 
                       {"religiousness":"somewhat"}, {"religiousness":"very"}])    
    
@app.route('/QuestionSix', methods=["GET", "POST"])
def QuestionSix():
    form_values.append(list(request.form.values()))
    return render_template('QuestionSix.html', 
    education_=[{"education":"grade school"}, {"education":"high school graduate"}, {"education":"some college"}, {"education":"college graduate"},
                   {"education":"some graduate work"}, {"education":"master's degree"}, {"education":"Ph.D, M.D., or other advanced degree"}]) 

@app.route('/QuestionSeven', methods=["GET", "POST"])
def QuestionSeven():
    form_values.append(list(request.form.values()))
    return render_template('QuestionSeven.html', 
    occupation_=[{"occupation":"unskilled employees"}, {"occupation":"machine operators and semiskilled employees"}, {"occupation":"skilled manual employees"}, 
        {"occupation":"clerical and sales workers, technicians, and owners of little businesses"}, {"occupation":"administrative personnel, owners of small businesses, and minor professionals"}, 
        {"occupation":"business managers, proprietors of medium-sized businesses, and lesser professionals"},
        {"occupation":"higher executives of large concerns, proprietors, and major professionals"}]) 

@app.route('/QuestionEight', methods=["GET", "POST"])
def QuestionEight():
    form_values.append(list(request.form.values()))
    return render_template('QuestionEight.html', 
    rating_=[{"marriage_rating":"very unhappy"}, {"marriage_rating":"somewhat unhappy"}, {"marriage_rating":"average"}, {"marriage_rating":"happier than average"}, 
        {"marriage_rating":"very happy"}]) 

@app.route('/QuestionNine', methods=["GET", "POST"])
def QuestionNine():
    form_values.append(list(request.form.values()))
    return render_template('QuestionNine.html', 
    num_affairs_=[{"num_affairs":"no affair"},{"num_affairs":"one affair"}, {"num_affairs":"two affairs"}, {"num_affairs":"three affairs"}, {"num_affairs":"four affairs"},
                     {"num_affairs":"five affairs"}, {"num_affairs":"three affairs"}, {"num_affairs":"seven affairs"}, {"num_affairs":"eight affairs"}, {"num_affairs":"nine affairs"},
                     {"num_affairs":"ten affairs"}, {"num_affairs":"eleven affairs"}, {"num_affairs":"twelve affairs or more"}])

@app.route('/NextPage', methods=["GET", "POST"])
def NextPage():
    form_values.append(list(request.form.values()))
    return render_template("NextPage.html")
    
    
input_dict={'[\'Yes\']': 1, '[\'No\']': 0, '[\'Male\']':1, '[\'Female\']':0, '[\'3 months or less\']': .125, '[\'4-6 months\']': .417, '[\'6 months\']': .75, '[\'1-2 years\']': 1.5, '[\'3-5 years\']': 4, 
'[\'6-8 years\']': 7, '[\'9-11 years\']': 10, '[\'12 or more years\']': 15, '[\'under 20\']': 17.5, '[\'20-24\']': 22, '[\'25-29\']': 27, '[\'30-34\']': 32, '[\'35-39\']': 37, 
'[\'40-44\']': 42, '[\'45-49\']': 47, '[\'50-54\']': 52, '[\'55 and over\']': 57, "[\'anti\']":1, "[\'not at all\']":2, "[\'slightly\']":3, "[\'somewhat\']":4, "[\'very\']":5, "[\'grade school\']":9, 
"[\'high school graduate\']":12, "[\'some college\']":14, "[\'college graduate\']":16, "[\'some graduate work\']":17, "[\"master\'s degree\"]":18, "[\'Ph.D, M.D., or other advanced degree\']":20,
"[\'unskilled employees\']":1, "[\'machine operators and semiskilled employees\']":2, "[\'skilled manual employees\']":3, "[\'clerical and sales workers, technicians, and owners of little businesses\']":4, 
"[\'administrative personnel, owners of small businesses, and minor professionals\']":5, "[\'business managers, proprietors of medium-sized businesses, and lesser professionals\']":6,
"[\'higher executives of large concerns, proprietors, and major professionals\']":7, "[\'very unhappy\']":1, "[\'somewhat unhappy\']":2, "[\'average\']":3, "[\'happier than average\']":4, "[\'very happy\']":5, 
"[\'no affair\']":0, "[\'one affair\']":1, "[\'two affairs\']":2, "[\'three affairs\']":3, "[\'four affairs\']":4, "[\'five affairs\']":5, "[\'three affairs\']":6, "[\'seven affairs\']":7, 
"[\'eight affairs\']":8, "[\'nine affairs\']":9, "[\'ten affairs\']":10, "[\'eleven affairs\']":11, "[\'twelve affairs or more\']":12, "[]":0}

					 
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    global form_values
    print(form_values)
    form_values=form_values[-9:]
    form_values_=[input_dict[str(i)] for i in form_values]
    
    form_val = np.array(form_values_).reshape(1, -1)
    prediction = model.predict(form_val)
    output = []
    if prediction[0]==1:
        output.append("Yes, you will likely have an affair in the future!")
    else:
        output.append("No, you are unlikely to have an affair in the future!")
    return render_template(
        'predict.html', prediction_text=f'Your selections were: \n  Gender: {[i[0] for i in form_values][0]} \n   Children: {[i[0] for i in form_values][1]}\
           \nYears married: {[i[0] for i in form_values][2]}  \n  Age group: {[i[0] for i in form_values][3]}  \n  Are you religious: {[i[0] for i in form_values][4]} \
           \nLevel of education: {[i[0] for i in form_values][5]}  \n  Occupation group: {[i[0] for i in form_values][6]} \n   Happiness in marriage: {[i[0] for i in form_values][7]} \
           \nNumber of affairs: {[i[0] for i in form_values][8]}\n' + str(output[0]))
        
					 
@app.route('/first_page')
def first_page():
    return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True, threaded=True)
