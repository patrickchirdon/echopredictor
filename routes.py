from flask import Flask, make_response, request, render_template, jsonify, Response
import io
from io import StringIO
import csv
import pandas as pd
import numpy as np
import pickle
import os
import json
from sqlalchemy import create_engine
import mysql.connector
from flask import Flask
import pandas as pd

import logging
import sys


engine=create_engine('sqlite://', echo=False)

app = Flask(__name__)




@app.route('/')
def form():
    return """
        <html>
        <head>
        <meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">

<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css" integrity="sha512-dTfge/zgoMYpP7QbHy4gWMEGsbsdZeCXz7irItjcC3sPUFtf0kuFbDz/ixG7ArTxmDjLXDmezHubeNikyKGVyQ==" crossorigin="anonymous">
        </head>
        <title>mass spec</title>
        <nav class="navbar navbar-inverse">
    <div class="container">
        <div class="navbar-header">
            <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar" aria-expanded="false" aria-controls="navbar">
                <span class="sr-only">Toggle navigation</span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
                <span class="icon-bar"></span>
            </button>
            <a class="navbar-brand" href="#">Biosortia Mass Spec Database Search</a>
        </div>
    </div>
</nav>


<div class="container">
            <body>
                <h1>Add To or Search Mass Spec Database</h1>
                </br>
                </br>
                <p> Insert a CSV file you would like to add to the database
                <form action="/transform" method="post" enctype="multipart/form-data">
                    <input type="file" name="data_file" class="btn btn-block"/>

                    <p>The CSV should have two columns labeled id and mw for compound id # and molecular weight </p>
                    <p>Search for compounds of interest:</p>
                    <p>Input the molecular weight of interest</p>
                    <p>lowest molecular weight in range</p>
                    <input type="text" id="lowmw" name="lowmw">
                    <p>highest molecular weight in range</p>
                    <input type="text" id="highmw" name="highmw">
                    <button type="submit" class="btn btn-success" value="Post comment">SearchDB</button>
                    <br>

                    <a href="https://github.com/patrickchirdon/massSpec">Source Code </a>
                    <br>
                    <h1>View Structural Alerts </h1>
                    <a href="https://datapane.com/reports/mA2oDG3/structural-alerts-copy-f4ut5ylb/">View Here</a>
                    <br>
                    <a href="https://medium.com/@patrickchirdon/mass-spec-random-forest-43819f1e6613">Help </a>


                    <div id="content"></div>







                </form>
                </div><!-- /.container -->
            </body>
        </html>
    """




@app.route('/transform', methods=["POST"])

def hello_world():
    """Call database and return data from df. Then display homepage."""
    try:



        myinput=transform_view()

        j=float(0)
        mylist=[]
        mylist2=[]
        for i in myinput['error']:
            mylist.append(i)
        for k in myinput['id']:
            mylist2.append(k)



        i=0

        for j in mylist:
            k=mylist2[i]
            add_signup_to_db(k, j)
            i=i+1



        #email_df = get_database_table_as_dataframe()


        mysearch=search()


        html_page = render_homepage(mysearch)
        return html_page
    except:
        logging.exception('Failed to connect to database.')

def render_homepage(df):
    """
    Note: you should use Flask's render_template to render HTML files.
    But here's a quick f-string HTML page that works:
    """
    result=df.to_html()

    mycsv=pd.read_csv('myvalues.csv')
    result2=mycsv.to_html()


    return result

def get_database_table_as_dataframe():
    """Connect to a table named 'Emails'. Returns pandas dataframe."""
    try:
        connection = mysql.connector.connect(
                            host='echoinvest.mysql.pythonanywhere-services.com',
                            db='echoinvest$tpyriformis',
                            user='echoinvest',
                            password='Zooboomafoo1@'
                            )

        email_df = pd.read_sql_query(sql="""SELECT * FROM massSpec""",
                               con=connection)
        logging.info(email_df.head())
        return email_df
    except:
        logging.exception('Failed to fetch dataframe from DB.')
        return "Oops!"
#endpoint for search


@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        lowmw = float(request.form['lowmw'])
        highmw= float(request.form['highmw'])


        connection = mysql.connector.connect(
                            host='echoinvest.mysql.pythonanywhere-services.com',
                            db='echoinvest$tpyriformis',
                            user='echoinvest',
                            password='Zooboomafoo1@'
                            )
        LOWMW=str(lowmw)
        HIGHMW=str(highmw)
        os.remove('myvalues.csv')
        mystring= "" +"SELECT id,mw FROM massSpec WHERE mw BETWEEN " + LOWMW + " AND " + HIGHMW +";" + ""
        email_df2 = pd.read_sql(sql=mystring,
                               con=connection)
        email_df2.to_csv('myvalues.csv')
        logging.info(email_df2.head())
        return email_df2
def transform_view():
    try:
        if request.method == 'POST':
            f = request.files['data_file']
        if not f:
            return "No file"


        stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
        thedf = pd.read_csv(stream, sep='\t')



    except:
        print('none')
    return thedf

@app.route("/add_signup_to_db", methods=["GET","POST"])
def add_signup_to_db(id, mw):
    """Pass data as SQL parameters with mysql."""
    try:
        connection = mysql.connector.connect(
                            host='echoinvest.mysql.pythonanywhere-services.com',
                            db='echoinvest$tpyriformis',
                            user='echoinvest',
                            password='Zooboomafoo1@'
                            )
        cursor = connection.cursor()
        sql = """INSERT INTO massSpec (id, mw) VALUES (%s, %s) """
        record_tuple = (id, mw)
        cursor.execute(sql,record_tuple)
        connection.commit()
    except mysql.connector.Error as error:
        logging.info("Failed to insert into MySQL table {}".format(error))
    except:
        logging.exception('Error inserting records to DB.')
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
        return("MySQL connection is closed")





if __name__ == "__main__":
    app.run(debug=True, port = 9000, host = "localhost")

