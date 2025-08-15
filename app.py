from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import hdbscan
import joblib
import os
import time
import warnings
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import folium
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from flask import Flask, render_template, request
import folium
import pandas as pd, numpy as np, tensorflow as tf
from folium.plugins import MarkerCluster


# Optional: suppress sklearn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
# --------- Load dataset ---------
data = pd.read_csv("CRIME.csv")
data['Area'] = data['Area'].astype(str).str.lower()
data['Ward'] = data['Ward'].astype(str).str.lower()
area_to_wards = data.groupby("Area")["Ward"].unique().apply(list).to_dict()

# Load models for prevention
tabnet_model = joblib.load("models/crime_prevention_tabnet.pkl")
scaler_strategy = joblib.load("models/scaler_strategy.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Load similarity bundle
with open('models/crime_similarity_bundle.pkl', 'rb') as f:
    bundle = pickle.load(f)

df = bundle['df']
similarity_matrix = bundle['similarity_matrix']
feature_cols = bundle['feature_cols']


# --------- HOME ---------
@app.route('/')
def home():
    return render_template('index.html')

# --------- CLUSTERING AND PREVENTION ---------
@app.route('/clustering', methods=['GET', 'POST'])
def clustering():
    result = None
    strategy = None
    plot_path = None
    wards = []
    selected_area = None

    if request.method == 'POST':
        area_input = request.form.get('area', '').strip().lower()
        ward_input = request.form.get('ward', '').strip().lower()
        crime_type = request.form.get('crime_type', '')
        description = request.form.get('description', '')
        selected_area = area_input

        if area_input in area_to_wards:
            wards = area_to_wards[area_input]
            if ward_input not in wards:
                return render_template('clustering.html', result={'error': 'Invalid ward selected for this area.'}, wards=wards, areas=list(area_to_wards.keys()), selected_area=area_input)

            matched_row = data[(data["Area"] == area_input) & (data["Ward"] == ward_input)]
            if matched_row.empty:
                result = {'error': "No matching Area and Ward found in dataset!"}
            else:
                lat, lon = matched_row.iloc[0]["Latitude"], matched_row.iloc[0]["Longitude"]
                coords = data[["Latitude", "Longitude"]].dropna()
                coords_index = coords.index

                scaler = StandardScaler()
                coords_scaled = scaler.fit_transform(coords)

                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                data.loc[coords_index, 'KMeans_Cluster'] = kmeans.fit_predict(coords_scaled)

                dbscan = DBSCAN(eps=0.2, min_samples=10)
                data.loc[coords_index, 'DBSCAN_Cluster'] = dbscan.fit_predict(coords_scaled)

                hdb = hdbscan.HDBSCAN(min_cluster_size=2)
                data.loc[coords_index, 'HDBSCAN_Cluster'] = hdb.fit_predict(coords_scaled)

                gmm = GaussianMixture(n_components=2, random_state=42)
                data.loc[coords_index, 'GMM_Cluster'] = gmm.fit_predict(coords_scaled)

                # Input location for clustering
                input_coords = scaler.transform(pd.DataFrame([[lat, lon]], columns=["Latitude", "Longitude"]))

                # Fit HDBSCAN on coordinates
                hdb_labels = hdb.fit_predict(coords_scaled)

                # Add input location
                all_coords = np.vstack([coords_scaled, input_coords])
                hdb_labels_with_input = hdb.fit_predict(all_coords)

                # Input location cluster label
                input_cluster_label = hdb_labels_with_input[-1]

                # Determine crime risk based on the input location's cluster size
                if input_cluster_label == -1:
                    crime_risk = "Low-Crime Area"
                else:
                    cluster_size = np.sum(hdb_labels_with_input == input_cluster_label)
                    if cluster_size >= 15:  # Threshold to adjust based on cluster size
                        crime_risk = "High-Crime Area"
                    else:
                        crime_risk = "Low-Crime Area"

                crime_distribution = matched_row["Primary Type"].value_counts(normalize=True) * 100
                primary_crime_type = crime_distribution.idxmax() if not crime_distribution.empty else "Unknown"

                arrest_rate = (matched_row["Arrest"].mean() * 100) if "Arrest" in matched_row else "Unknown"
                poverty_rate = matched_row["Poverty Rate"].values[0] if "Poverty Rate" in matched_row else "Unknown"
                education_level = matched_row["Education Level"].values[0] if "Education Level" in matched_row else "Unknown"
                job_opportunities = matched_row["Job Opportunities"].values[0] if "Job Opportunities" in matched_row else "Unknown"

                poverty_desc = "High" if poverty_rate >= 2 else "Low"
                education_desc = "Low" if education_level <= 1 else "High"
                job_opportunities_desc = "Few" if job_opportunities <= 1 else "Many"
                distance_to_hotspot = matched_row.filter(like='Distance_to_Hotspot').min(axis=1).values[0]

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(data["Longitude"], data["Latitude"], c='lightblue', label="Crime Locations")
                ax.scatter(lon, lat, color='red', marker='X', s=100, label="Input Location")
                ax.set_title("Crime Cluster Visualization")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
                ax.legend()
                plot_path = "static/plot.png"
                plt.savefig(plot_path)
                plt.close()

                result = {
                    "area": area_input.title(),
                    "ward": ward_input,
                    "crime_risk": crime_risk,
                    "primary_crime_type": primary_crime_type,
                    "arrest_rate": f"{arrest_rate:.2f}%" if isinstance(arrest_rate, float) else "Unknown",
                    "poverty": poverty_desc,
                    "education": education_desc,
                    "jobs": job_opportunities_desc,
                    "hotspot_distance": f"{distance_to_hotspot:.2f} km",
                    'total_crimes': matched_row["Total_Crimes_In_Area"].values[0],
                    'is_violent': matched_row["Is_Violent_Crime"].values[0],
                    'arrest': matched_row["Arrest"].values[0],
                    'poverty_index': matched_row["Poverty_Unemployment_Index"].values[0],
                    'stress_index': matched_row["Socioeconomic_Stress_Index"].values[0],
                    'is_night': matched_row["Is_Night"].values[0],
                    'is_weekend': matched_row["Is_Weekend"].values[0]
                }

                # Predict strategy without Weather feature
                structured_input = np.array([[
                    result['total_crimes'],
                    result['is_violent'],
                    result['arrest'],
                    result['poverty_index'],
                    result['stress_index'],
                    result['is_night'],
                    result['is_weekend']
                ]])

                tfidf_input = tfidf_vectorizer.transform([crime_type + " " + description]).toarray()
                final_input = np.hstack((structured_input, tfidf_input))
                final_input = scaler_strategy.transform(final_input)

                pred = tabnet_model.predict(final_input)[0]
                strategy_mapping = {
                    0: "Increase Police Presence",
                    1: "General Crime Awareness Campaign",
                    2: "General Crime Awareness Campaign",
                    3: "Increase Police Presence",
                    4: "Community Outreach Programs"
                }

                strategy = strategy_mapping.get(int(pred), "Unknown Strategy")

    return render_template(
        'clustering.html',
        result=result,
        plot_path=plot_path,
        strategy=strategy,
        wards=wards,
        areas=list(area_to_wards.keys()),
        selected_area=selected_area
    )

# --------- SIMILARITY ---------
with open("models/crime_similarity_bundle.pkl", "rb") as f:
    loaded_bundle = pickle.load(f)

df_loaded = loaded_bundle['df']
similarity_matrix_loaded = loaded_bundle['similarity_matrix']
filter_config_loaded = loaded_bundle['filter_config']

@app.route('/similarity')
def similarity_page():
    return render_template(
        'similarity.html',
        crime_snos=filter_config_loaded['crime_sno_options'],
        primary_types=filter_config_loaded['crime_type_options'],
        areas=filter_config_loaded['area_options'],
        quarters=filter_config_loaded['quarter_options'],
        days=filter_config_loaded['day_options'],
        hours=filter_config_loaded['hour_options'],
        severities=filter_config_loaded['severity_options']
    )

@app.route('/find_similar', methods=['POST'])
def find_similar():
    data_form = request.get_json()

    crime_sno = int(data_form.get("crime_sno"))
    area = data_form.get("area")
    quarter = data_form.get("quarter")
    day = data_form.get("day")
    hour = data_form.get("hour")
    top_n = int(data_form.get("top_n"))

    if crime_sno not in df_loaded.index:
        return jsonify({"error": "❌ Selected Crime SNo does not exist in the dataset."})

    filtered_df = df_loaded.copy()

    if area != 'All':
        filtered_df = filtered_df[filtered_df['Area'] == area]
    if quarter != 'All':
        filtered_df = filtered_df[filtered_df['Quarter'] == int(quarter)]
    if day != 'All':
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == int(day)]
    if hour != 'All':
        filtered_df = filtered_df[filtered_df['Hour'] == int(hour)]

    filtered_indices = filtered_df.index.tolist()

    if not filtered_indices:
        closest_indices = np.argsort(-similarity_matrix_loaded[crime_sno])[:top_n]
        closest_results = df_loaded.loc[closest_indices, [
            'Primary Type', 'Description', 'Hour', 'DayOfWeek', 'Quarter', 'Area'
        ]]
        closest_results.insert(0, 'Crime SNo', closest_indices)  # Add Crime SNo column
        return jsonify({
            "message": "⚠️ No crimes matched the filters. Showing closest overall matches.",
            "results": closest_results.to_dict(orient="records")
        })

    if crime_sno not in filtered_indices:
        crime_sno = filtered_indices[0]

    filtered_index_map = {idx: i for i, idx in enumerate(filtered_indices)}
    similarity_subset = similarity_matrix_loaded[crime_sno, filtered_indices]
    sorted_indices = np.argsort(-similarity_subset)[:top_n]
    selected_indices = [filtered_indices[i] for i in sorted_indices]

    result = filtered_df.loc[selected_indices, [
        'Primary Type', 'Description', 'Hour', 'DayOfWeek', 'Quarter', 'Area'
    ]]
    result.insert(0, 'Crime SNo', selected_indices)  # Add Crime SNo column

    return jsonify({
        "message": "✅ Found similar crimes.",
        "results": result.to_dict(orient="records")
    })
@app.route("/forecast")
def forecast():
    with open("models/clean_crime_forecast_model.pkl", "rb") as f:
        data = pickle.load(f)

    predictions = data["predictions"]
    label_classes = data["label_classes"]

    def get_severity_color(type_vector):
        dominant_type = label_classes[np.argmax(type_vector)]

        high_crimes = [
            "CONCEALED CARRY LICENSE VIOLATION", "CRIM SEXUAL ASSAULT", "HOMICIDE",
            "HUMAN TRAFFICKING", "KIDNAPPING", "NARCOTICS", "OFFENSE INVOLVING CHILDREN", "SEX OFFENSE"
        ]
        medium_crimes = [
            "ARSON", "BURGLARY", "BATTERY", "DECEPTIVE PRACTICE", "CRIMINAL DAMAGE",
            "CRIMINAL TRESPASS", "INTERFERENCE WITH PUBLIC OFFICER", "INTIMIDATION",
            "LIQUOR LAW VIOLATION", "MOTOR VEHICLE THEFT", "OBSCENITY",
            "PROSTITUTION", "ROBBERY", "THEFT", "WEAPONS VIOLATION"
        ]

        if dominant_type in high_crimes:
            return "red"
        elif dominant_type in medium_crimes:
            return "orange"
        else:
            return "green"

    # Generate the map
    lat_center = np.mean([pred["Latitude"] for pred in predictions])
    lon_center = np.mean([pred["Longitude"] for pred in predictions])
    crime_map = folium.Map(location=[lat_center, lon_center], zoom_start=11)

    for pred in predictions:
        lat, lon = pred["Latitude"], pred["Longitude"]
        for date, count, type_vector in zip(pred["Dates"], pred["Counts"], pred["Types"]):
            if count <= 0:
                continue

            crime_type = label_classes[np.argmax(type_vector)]
            popup_content = f"""
                📍 <b>Crime Forecast</b><br>
                📅 {date.strftime('%d-%m-%Y')}<br>
                🔢 Predicted: {count:.2f}<br>
                🔍 Top Type: {crime_type}<br>
                📍 Location: Latitude: {lat}, Longitude: {lon}
            """
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_content, max_width=250),
                icon=folium.Icon(color=get_severity_color(type_vector))
            ).add_to(crime_map)

    crime_map.save("templates/map.html")
    return render_template("map.html")

#SEVERITY

try:
    df_filtered = joblib.load(open("models/severity_data.pkl", "rb"))
    label_encoder = joblib.load(open("models/label_encoder_severity.pkl", "rb"))
except Exception as e:
    print("Error loading model files:", e)
    df_filtered = None
    label_encoder = None

@app.route("/severity", methods=["GET", "POST"])
def severity():
    crime_map_html = None
    min_val, max_val = 20, 80  # default slider values

    if request.method == "POST":
        try:
            min_val = int(request.form.get("min_severity", 20))
            max_val = int(request.form.get("max_severity", 80))
        except ValueError:
            min_val, max_val = 20, 80  # fallback to defaults if invalid input

        if df_filtered is not None:
            min_score = min_val / 100
            max_score = max_val / 100

            filtered_data = df_filtered[
                (df_filtered["Severity_Score"] >= min_score) &
                (df_filtered["Severity_Score"] <= max_score)
            ]

            crime_map = folium.Map(
                location=[df_filtered["Latitude"].mean(), df_filtered["Longitude"].mean()],
                zoom_start=11
            )
            marker_cluster = MarkerCluster().add_to(crime_map)

            for _, row in filtered_data.iterrows():
                try:
                    crime_type = label_encoder.inverse_transform([int(row["Primary Type"])]).item()
                except:
                    crime_type = "Unknown"

                folium.Marker(
                    location=[row["Latitude"], row["Longitude"]],
                    popup=folium.Popup(f"""
                        <b>Type:</b> {crime_type}<br>
                        <b>Severity:</b> {row['Severity_Score'] * 100:.0f}%
                    """, max_width=300),
                    icon=folium.Icon(color="blue")
                ).add_to(marker_cluster)

            crime_map_html = crime_map._repr_html_()

    return render_template("severity.html", crime_map=crime_map_html, min_val=min_val, max_val=max_val)

if __name__ == '__main__':
    app.run(debug=True)
