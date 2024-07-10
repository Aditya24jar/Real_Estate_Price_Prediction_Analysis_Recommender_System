from flask import Flask,render_template,request,url_for,redirect,jsonify
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import base64

app =  Flask(__name__)

with open('df.pkl','rb') as file:
    df = pickle.load(file)

with open('pipeline.pkl','rb') as file:
    pipeline = pickle.load(file)

def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3

    # Get the similarity scores for the property using its name as the index
    sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))

    # Sort properties based on the similarity scores
    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices and scores of the top_n most similar properties
    top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

    # Retrieve the names of the top properties using the indices
    top_properties = location_df.index[top_indices].tolist()

    # Create a dataframe with the results
    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df



location_df = pickle.load(open('location_distance.pkl','rb'))
cosine_sim1 = pickle.load(open('cosine_sim1.pkl','rb'))
cosine_sim2 = pickle.load(open('cosine_sim2.pkl','rb'))
cosine_sim3 = pickle.load(open('cosine_sim3.pkl','rb'))

sectors = sorted(df['sector'].unique().tolist())
bedRooms = sorted(df['bedRoom'].unique().tolist())
bathRooms = sorted(df['bathroom'].unique().tolist())
balconies = sorted(df['balcony'].unique().tolist())
property_ages = sorted(df['agePossession'].unique().tolist())
furnishing_types = sorted(df['furnishing_type'].unique().tolist())
luxury_categories = sorted(df['luxury_category'].unique().tolist())
floor_categories = sorted(df['floor_category'].unique().tolist())


@app.route('/')
def home():
    return render_template('index.html', 
                           sectors=sectors, 
                           bedRooms=bedRooms, 
                           balconies=balconies, 
                           bathRooms = bathRooms,
                           property_ages=property_ages, 
                           furnishing_types=furnishing_types, 
                           luxury_categories=luxury_categories, 
                           floor_categories=floor_categories)


@app.route('/predict', methods=['POST'])
def predict():
  
    property_type = request.form.get('Property_Type')
    sector = request.form.get('Sector')
    number_of_bedrooms = float(request.form.get('NumberOfBedrooms'))
    number_of_bathrooms = float(request.form.get('NumberOfBathrooms'))
    number_of_balconies = request.form.get('balcony')
    property_age = request.form.get('property_age')
    built_up_area = float(request.form.get('built_up_area'))
    servant_room = float(request.form.get('servant_room'))
    store_room = float(request.form.get('store_room'))
    furnishing_type = request.form.get('furnishing_type')
    luxury_category = request.form.get('luxury_category')
    floor_category = request.form.get('floor_category')

    data = [[property_type, sector, number_of_bedrooms, number_of_bathrooms, number_of_balconies, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
            'agePossession', 'built_up_area', 'servant room', 'store room',
            'furnishing_type', 'luxury_category', 'floor_category']
        
    one_df = pd.DataFrame(data, columns=columns)

    base_price = float(np.expm1(pipeline.predict(one_df))[0])
    low = base_price - 0.22
    high = base_price + 0.22

    return render_template('index.html', lower = low, upper = high, sectors=sectors, 
                           bedRooms=bedRooms, 
                           balconies=balconies, 
                           bathRooms = bathRooms,
                           property_ages=property_ages, 
                           furnishing_types=furnishing_types, 
                           luxury_categories=luxury_categories, 
                           floor_categories=floor_categories)

@app.route('/analyse')
def analysed():
    new_df = pd.read_csv('data_viz1.csv')
    feature_text = pickle.load(open('feature_text.pkl','rb'))

    group_df = new_df.groupby('sector')[['price','price_per_sqft','built_up_area','latitude','longitude']].mean()

    fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
                    color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                    mapbox_style="open-street-map",width=1080,height=600,hover_name=group_df.index)
    
    # Set the title of the plot
    fig.update_layout(title='Sector Price per sqft Geomap')
    
    graph = plotly.io.to_json(fig)

    # Read the saved image file and encode it to base64
    with open('wordcloud.png', 'rb') as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')

    fig1 = px.scatter(new_df[new_df['property_type'] == 'house'], x="built_up_area", y="price", color="bedRoom", title="Area Vs Price for house")


    fig2 = px.scatter(new_df[new_df['property_type'] == 'flat'], x="built_up_area", y="price", color="bedRoom",
                        title="Area Vs Price for flat")
    
    graph1 = plotly.io.to_json(fig1)
    graph2 = plotly.io.to_json(fig2)

    fig3 = px.pie(new_df, names='bedRoom')
    graph3 = plotly.io.to_json(fig3)

    fig4 = px.box(new_df[new_df['bedRoom'] <= 6], x='bedRoom', y='price', title='BHK Price Range upto 6 bedrooms')
    graph4 = plotly.io.to_json(fig4)

    with open('distribution.png', 'rb') as f1:
        image_data1 = f1.read()
    encoded_image1 = base64.b64encode(image_data1).decode('utf-8')
    return render_template('analytics.html', plot=graph, distribution = encoded_image1, wordcloud=encoded_image, plot1 = graph1, plot2 = graph2, plot3 = graph3, plot4 = graph4)

@app.route('/base')
def base():
    locations = sorted(location_df.columns.to_list())
    apartments = sorted(location_df.index.to_list())
    return render_template('recommender.html', locations=locations, apartments=apartments)


@app.route('/search', methods=['POST'])
def search():
    selected_location = request.form['location']
    radius = float(request.form['radius'])
    result_ser = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()
    results = {key: round(value / 1000) for key, value in result_ser.items()}
    return render_template('recommender.html', locations=sorted(location_df.columns.to_list()), 
                           apartments=sorted(location_df.index.to_list()), 
                           results=results)
    
@app.route('/recommend', methods=['POST'])
def recommend():
    selected_apartment = request.form['apartment']
    recommendation_df = recommend_properties_with_scores(selected_apartment)
    recommendations = recommendation_df.to_dict(orient='records')
    return render_template('recommender.html', locations=sorted(location_df.columns.to_list()), 
                           apartments=sorted(location_df.index.to_list()), 
                           recommendations=recommendations)


if __name__ == '__main__':
    app.run(debug = True)