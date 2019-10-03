from flask import Flask,render_template,request,redirect,url_for,session,send_file
from werkzeug import secure_filename
from flask_session import Session
import data_process as dp
import numpy as np
#import random
#from time import sleep
#from flask_socketio import SocketIO,emit
app = Flask(__name__)

@app.route('/')
def main_page():
    result={'file_uploaded':False}
    if('dataset' in session.keys()):
        result['file_uploaded']=True
    return render_template('home.html',result=result)

@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/get_file',methods = ['POST', 'GET'])
def get_file():
    if (request.method == 'POST'):
        data=request.files['data']
        data.save(secure_filename(data.filename))
        session['dataset']=data.filename
        session['index'], session['preprocessed_data'], session['selected_bol'], session['df'], session['col_name'] = dp.prepare(session['dataset'])
        return redirect(url_for('main_page'))

@app.route('/plots', methods = ['POST', 'GET'])
def plots():
    result={}
    result['have_img'] = False
    if(request.method == 'POST'):
        if('values_1' in (request.form).keys()):
            result['sensor_name'],result['data'] = dp.sensor_plot(session['col_name'],request.form.getlist('values_1'), session['df'])
        else:
            result['sensor_name'],result['data'] = dp.sensor_plot(session['col_name'],request.form.getlist('values_2'), session['df'])
        result['have_img']=True
        result['index']= session['index']
    result['col_name']=session['col_name']
    result['selected_bol']=session['selected_bol']
    return render_template('plots.html',result=result)


@app.route('/train_model', methods=['POST', 'GET'])
def train_model():
    result={}
    result['have_img'] = False
    if request.method == 'POST':
        if('batch_size' in (request.form).keys()):
            result['data'], result['split'] = dp.model_create(session['preprocessed_data'],request.form['batch_size'],request.form['n_z'],request.form['epochs'],request.form['split'])
            session['predicted_data']= dp.predict_on_data(session['preprocessed_data'])
            session['split']=result['split']
        #else:
            #result['sensor_name'],result['data'] = dp.sensor_plot(session['col_name'],request.form.getlist('values_2'), session['df'])
        result['have_img']=True
    return render_template('train_model.html',result=result)


@app.route('/train_plots')
def train_plots():
    result={}
    session['loss']= dp.loss(session['preprocessed_data'], session['predicted_data'])
    result['test_loss']=(session['loss'][:session['split']]).tolist()
    result['predicted_loss']=(session['loss'][session['split']:]).tolist()
    result['test_index']=session['index'][:session['split']]
    result['pred_index']=session['index'][session['split']:]
    result['split']=session['split']
    result['have_img']=True
    return render_template('train_plots.html',result=result)

@app.route('/variable_adjustment', methods=['POST', 'GET'])
def variable_adjustment():
    result={}
    result['have_img']=False
    result['max']=0
    if request.method == 'POST':
        if('thresh_loss' in (request.form).keys()):
            if(request.form['method_used']=="method1"):
                session['lis'],session['max'],session['anamoly_index']=dp.anamoly_calc(session['loss'],session['preprocessed_data'], request.form['thresh_loss'],request.form['window_size'])
                result['max']=session['max']
            if (request.form['method_used'] == "method2"):
                session['lis'], session['max'], session['anamoly_index'] = dp.anamoly_calc2(session['loss'],session['preprocessed_data'],request.form['thresh_loss'],session['predicted_data'])
                result['max'] = session['max']
        if('thresh_percen' in (request.form).keys()):
            session['thresh_percen']=request.form['thresh_percen']
            result['max'] = session['max']
            result['have_img']=True
            result['loss']=session['loss'].tolist()
            result['index']=session['index']
            result['anamoly_index']=session['index'][session['anamoly_index']]
            result['anamoly_loss']=session['loss'][session['anamoly_index']].tolist()
            session['ana_sensors'], session['ana_per'] = dp.final_calc(session['thresh_percen'], session['lis'],session['col_name'], session['selected_bol'])
    return render_template('var_adjustment.html',result=result)


@app.route('/view_anamoly', methods=['POST', 'GET'])
def view_anamoly():
    result={}
    result['have_table']=False
    result['ana_index'] = session['index'][session['anamoly_index']]
    if request.method == 'POST':
        val=int(request.form['value'])
        result['ana_sensors']=session['ana_sensors'][val]
        result['ana_per'] = session['ana_per'][val]
        result['have_table']=True
    return render_template('view_anamoly.html',result=result)


@app.route('/act_pred', methods=['POST', 'GET'])
def act_pred():
    result={}
    result['sensors']=(np.asarray(session['col_name'])[session['selected_bol']]).tolist()
    result['actual']=(session['preprocessed_data'].transpose()).tolist()
    result['pred']=(session['predicted_data'].transpose()).tolist()
    result['index']=session['index']
    result['split'] = session['index'][session['split']]
    result['split1'] = session['index'][-1]
    #result['anamoly_index'] = session['index'][session['anamoly_index']]
    #print(result['sensors'])
    #print(session['preprocessed_data'].shape)
    #print(session['predicted_data'].shape)
    #print(result['pred'])
    #print(result['actual'])
    return render_template('act_pred.html',result=result)


if __name__ == '__main__':
    SESSION_TYPE = 'filesystem'
    app.config.from_object(__name__)
    Session(app)
    app.run(debug=True)