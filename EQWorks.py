#!/usr/bin/env python
# coding: utf-8

# In[19]:


# Library Imports 
from pyspark.conf import SparkConf               # Spark Configuraton
from pyspark.sql import SparkSession             # Spark Session
import pyspark.sql.functions as F                # For SQL Functions
from pyspark.sql import Window                   # For Applying Window Functions
from mpl_toolkits.basemap import Basemap         # Plotting GeoSpatial Visualization
import numpy as np                               # Numpy
import seaborn as sns                            # SeaBorn visualizations
import matplotlib.pyplot as plt                  # Matplot visualizations
import pandas as pd                              # For Pandas Dataframes
from pylab import rcParams                       # rcParams
import sys

# Setting up the Spark session 
spark = SparkSession.builder.config(conf=SparkConf()).getOrCreate()


# In[2]:


def remove_outliers(df,threshold=3):
    """ Spark Function to remove outliers.
    This function applies a window over POI Id to filter
    the rows which are n standard deviation away from the mean for each group
    
    Parameters df: DataFrame
               threshold(integer): Number of Standard deviations 
    """
    # Apply window function over POI ID and filter out outliers
    w = Window.partitionBy('POIID')
    df = df.withColumn('OutlierLimit',threshold*F.stddev(F.col('distance')).over(w) + F.avg(F.col('distance')).over(w))    .where(F.col('distance') <= F.col('OutlierLimit'))    .drop('OutlierLimit')
    return df


# In[3]:


def minmaxscaler(x,lower_interval,upper_interval):
    """ Scales values between the lower and upper limit
    Parameters x: List to be scaled
               lower_interval: lower limit value
               upper_interval: upper limit value
    """
    minimum = np.min(x)
    maximum = np.max(x)
    # Range Normalization Equation 
    scaled = [((upper_interval - lower_interval)*(k - minimum)/(maximum - minimum) + lower_interval) for k in x]
    return scaled


# In[4]:


def process_data_sample(filepath):
    """ Reads the requests data and removes redundant entries
    
    Parameters filepath: path to the file to be read
    """
    # Read the data frame
    df_sample = spark.read.csv(filepath, header=True, mode="DROPMALFORMED")
    # Caches the spark dataframe into memory
    df_sample.cache()
    # Renames the column as it contains additional space
    df_sample = df_sample.withColumnRenamed(' TimeSt', 'TimeSt')
    # Concatenates TimeSt, Latitude and Longitude as per requirement 
    df_sample = df_sample.withColumn('GeoSpatialInfo',F.concat(df_sample.TimeSt, df_sample.Latitude,df_sample.Longitude))
    # Removes duplicates 
    df_sample = df_sample.drop_duplicates(subset=['GeoSpatialInfo']).drop('GeoSpatialInfo')
    return df_sample


# In[5]:


def process_poi_list(filepath):
    """ Reads the POI data and removes redundant entries
    Parameters filepath: path to the file to be read
    """
    # Read the data frame
    df_poi = spark.read.csv(filepath, header=True, mode="DROPMALFORMED")
    # Caches the spark dataframe into memory
    df_poi.cache()
    # Renames the Latitude and Longitude columns to distinguish between request and POI coordinates
    df_poi = df_poi.withColumnRenamed(' Latitude', 'POI_Latitude').withColumnRenamed('Longitude','POI_Longitude')
    # Concatenates Lat and Lon to identify duplicates
    df_poi = df_poi.withColumn('LatLon',F.concat(df_poi.POI_Latitude,df_poi.POI_Longitude))
    # Removes Duplicates
    df_poi = df_poi.drop_duplicates(subset=['LatLon']).drop('LatLon')
    return df_poi


# In[6]:


def join_and_analyze(df_poi,df_sample):
    """ Joins the Requests data and POI list data, calculates distance between POI Centers
    and retains the record with the minimum distance to a particular POI center
    
    Parameters: df_poi: POI List datafarme 
                df_sample: Requests dataframe
    
    """
    # Since there are no matching fields between the data, cartesian product is done to combine the datasets
    df_joined = df_sample.crossJoin(df_poi)
    # Caching to memory
    df_joined.cache()
    # Applying the Haversine formula to determine distance between coordinate pairs
    df_joined = df_joined.withColumn("a", (
    F.pow(F.sin(F.radians(F.col("POI_Latitude") - F.col("Latitude")) / 2), 2) +
    F.cos(F.radians(F.col("Latitude"))) * F.cos(F.radians(F.col("POI_Latitude"))) *
    F.pow(F.sin(F.radians(F.col("POI_Longitude") - F.col("Longitude")) / 2), 2)
    )).withColumn("distance", F.atan2(F.sqrt(F.col("a")), F.sqrt(-F.col("a") + 1)) * 2 * 6371)
    
    # Applying window function to retain the records with the least distance to a POI center
    w = Window.partitionBy('_ID')
    df_joined = df_joined.withColumn('min', F.min('distance').over(w))    .where(F.col('distance') == F.col('min'))    .drop('min').drop('a')

    return df_joined


# In[7]:


def calculate_mean_stdv(df_joined):
    """
    Applies Aggregate function to calculate the Mean and Standard Deviation of the POI centers
    Parameters: df_joined: DataFrame combining requests and POI data
    """
    df_summary = df_joined.groupBy(F.col("POIID")).agg(F.avg(F.col("distance")).alias("AvgDistance"), F.stddev(F.col("distance")).alias("standDev"))
    df_summary = df_summary.toPandas()
    return df_summary


# In[8]:


def mathematical_model(df,poi):
    """ Mathematical model for determining the popularity of the three POIs
    Based on the understanding of the data and readings from research papers
    such as http://www.vldb.org/pvldb/vol10/p1010-liu.pdf, the model was designed
    using the distance and temporal measures, it returns the mean score for each POI
    Parameters: df:  Main Dataframe
                poi: POI Label"""
    score = []
    # Filter on POI 
    df_poi = df[df.POIID==poi]
    # Sorting the dataframe
    df_poi = df_poi.sort_values(by="distance").reset_index(drop=True)
    # Model Equation
    score.extend((1/np.power(df_poi.distance,2))*np.power(df_poi.Time_Metric,2))
    # Returning the mean score
    return np.mean(score)


# In[9]:


def calculate_radius_and_density(df_joined):
    """
    Applies Aggregate function to calculate the Radius and Density of each POI
    Parameters: df_joined:DataFrame combining requests and POI data

    """
    df_summary = df_joined.groupBy(F.col("POIID")).agg(F.max(F.col("distance")).alias("Radius"),((F.count(F.col("_ID")))/(np.pi*np.power(F.max(F.col("distance")),2))).alias("Density( Request/Area)"))
    return df_summary


# In[10]:


def calculate_new_coordinates(lat1, lon1, rad, dist):
    """
    Calculate coordinate pair given starting point, radial and distance
    Method from: http://www.geomidpoint.com/destination/calculation.html
    
    This code was borrowed from Stackoverflow and is based on the above link
    """

    flat = 298.257223563
    a = 2 * 6378137.00
    b = 2 * 6356752.3142

    # Calculate the destination point using Vincenty's formula
    f = 1 / flat
    sb = np.sin(rad)
    cb = np.cos(rad)
    tu1 = (1 - f) * np.tan(lat1)
    cu1 = 1 / np.sqrt((1 + tu1*tu1))
    su1 = tu1 * cu1
    s2 = np.arctan2(tu1, cb)
    sa = cu1 * sb
    csa = 1 - sa * sa
    us = csa * (a * a - b * b) / (b * b)
    A = 1 + us / 16384 * (4096 + us * (-768 + us * (320 - 175 * us)))
    B = us / 1024 * (256 + us * (-128 + us * (74 - 47 * us)))
    s1 = dist / (b * A)
    s1p = 2 * np.pi

    while (abs(s1 - s1p) > 1e-12):
        cs1m = np.cos(2 * s2 + s1)
        ss1 = np.sin(s1)
        cs1 = np.cos(s1)
        ds1 = B * ss1 * (cs1m + B / 4 * (cs1 * (- 1 + 2 * cs1m * cs1m) - B / 6 *             cs1m * (- 3 + 4 * ss1 * ss1) * (-3 + 4 * cs1m * cs1m)))
        s1p = s1
        s1 = dist / (b * A) + ds1

    t = su1 * ss1 - cu1 * cs1 * cb
    lat2 = np.arctan2(su1 * cs1 + cu1 * ss1 * cb, (1 - f) * np.sqrt(sa * sa + t * t))
    l2 = np.arctan2(ss1 * sb, cu1 * cs1 - su1 * ss1 * cb)
    c = f / 16 * csa * (4 + f * (4 - 3 * csa))
    l = l2 - (1 - c) * f * sa * (s1 + c * ss1 * (cs1m + c * cs1 * (-1 + 2 * cs1m * cs1m)))
    d = np.arctan2(sa, -t)
    finaltc = d + 2 * np.pi
    backtc = d + np.pi
    lon2 = lon1 + l

    return (np.rad2deg(lat2), np.rad2deg(lon2))


# In[11]:


def shaded_great_circle(m, lat_0, lon_0, dist=100, col='k'):
    """
    This method uses the calculate_new_coordinates method to map out the latitudes and longitudes 
    for drawing the circle given a radius on a map
    Parameters: m: Basemap
                lat_0: Latitude of POI center
                lon_0: Longitude of POI center
                dist: Radial distance (or radius)
                col: colour
    """
    # Distance conversion
    dist = dist * 2000
    # Converts POI centers to Radians
    theta_arr = np.linspace(0, np.deg2rad(360), 100)
    lat_0 = np.deg2rad(lat_0)
    lon_0 = np.deg2rad(lon_0)

    coords_new = []
    # Determines the coordinates of the circle
    for theta in theta_arr:
        coords_new.append(calculate_new_coordinates(lat_0, lon_0, theta, dist))

    lat = [item[0] for item in coords_new]
    lon = [item[1] for item in coords_new]
    # Maps to basemap
    x, y = m(lon, lat)
    # Plotting
    m.plot(x, y, col)


# In[12]:


def visualize_requests(df,df_poi):
    """
    This method plots the visualization of data points mapped to POI clusters
    Parameters df: Requests DataFrame
               df_poi: POI Data
    """
    # Converting String fields to float for visualization purposes
    df_poi.POI_Latitude = df_poi["POI_Latitude"].astype(float)
    df_poi.POI_Longitude = df_poi["POI_Longitude"].astype(float)
    df_poi.Radius = df_poi["Radius"].astype(float)
    
    # Colours list
    colours_circle = ['green','red','blue']
    
    # setup Mercator map projection for Canadian Map
    m = Basemap(projection = "merc", resolution = "l",area_thresh = 100,
               llcrnrlon = -143, llcrnrlat = 42, urcrnrlon = -50, urcrnrlat = 79,lat_0=58.747544,lon_0=-104.239655)

    # Setting Latitudes and longitudes of Point of interest
    coords = dict()
    for i in range(len(df_poi)):
        coords[df_poi.loc[i,"POIID"]] = [df_poi.loc[i,"POI_Latitude"], df_poi.loc[i,"POI_Longitude"]]
    
    # Setting Figsize
    plt.figure(figsize=(16,16))
    # Plot markers and labels on map
    k = 0
    for key in coords:
        lat, lon = coords[key]
        x,y = m(float(lon), float(lat))
        # Plotting Cluster and setting a Label
        m.plot(x, y, 'bo', markersize=5)
        plt.text(x+10000, y+5000, key, color='black',fontsize = 15)


    # For coastlines
    m.drawcoastlines()
    # Filling Continent 
    m.fillcontinents(color='pink',lake_color='aqua')
    # Filling Map Boundaries
    m.drawmapboundary(fill_color='aqua')



    # Repeat for Each POI
    for i in range(len(df_poi)):
        # Draw the circle
        shaded_great_circle(m, df_poi.loc[i,"POI_Latitude"], df_poi.loc[i,"POI_Longitude"], df_poi.loc[i,"Radius"], col=colours_circle[i])
        # Filter based on POI Label
        df_new = df[df.POIID == df_poi.loc[i,"POIID"]]
        # Map the Basemap with points of the particular POI
        x,y = m(df_new.Longitude.astype(float).values, df_new.Latitude.astype(float).values)
        # Plot the values
        m.plot(x,y,'bo', color=colours_circle[i],markersize=4)
    # Show the figure
    plt.show()


# In[13]:


def perform_eda(df):
    """
    Some Exploratory Analysis done, I have not called this method, this is for reference
    
    Parameters df: Requests Dataset
    """
    plt.hist(df.distance, bins="auto")
    sns.boxplot(x="POIID", y='distance', data=df)
    df.City[df.POIID == 'POI3'].value_counts(sort= True).head(10).plot(kind="bar")
    df.City[df.POIID == 'POI1'].value_counts(sort= True).head(10).plot(kind="bar")
    df.City[df.POIID == 'POI4'].value_counts(sort= True).head(10).plot(kind="bar")


# In[14]:


def build_time_metric(df_joined,dictionary):
    """
    Time Metric designed using the time_score dictionary
    Parameters df_joined: Requests Dataframe
               dictionary: time_score dictionary for mapping
    """
    # Extract the hour
    df_joined["Time"] = df_joined["TimeSt"].str[11:13]
    # Map the score
    df_joined["Time_Metric"] = df_joined["Time"].map(dictionary)
    return df_joined


# In[15]:


def run_model(df_requests):
    """Runs the Mathematical model for each POI, applies the scale and consolidates the results
    into a summary table
    Parameters: df_requests: Requests DataFrame"""
    
    # Get the list of POI IDs
    poi_list = df_requests.POIID.unique()
    
    # Initializing the list
    results = []
    # Loop through each POI
    for i in poi_list:
        # Apply the Mathematical model
        results.append(mathematical_model(df_requests,poi = i))
    # Scale the results
    scaled = minmaxscaler(results,-10,10)
    # Build the dataframe
    df_results = pd.DataFrame()
    df_results = df_results.assign(POIID=poi_list)
    df_results = df_results.assign(Score=scaled)
    return df_results
    


# In[16]:


# Define metrics Time Metrics, based on peak and off peak hours
# Custom made dictionary based on intiution and readings
time_score = {"00":0.75,
        "01":0.75,
        "02":0.75,
        "03":0.75,
        "04":0.75,
        "05":0.75,
        "06":0.75,
        "07":1,
        "08":1.5,
        "09":1.5,
        "10":1.5,
        "11":1,
        "12":1,
        "13":1,
        "14":1,
        "15":1,
        "16":1,
        "17":1.5,
        "18":2,
        "19":2,
        "20":2,
        "21":1,
        "22":1,
        "23":0.75}


# In[28]:


def main(visualize=True):
    "Main Function"
    

    # Get Requests Data
    if sys.argv[1]:
        df_sample = process_data_sample(filepath=sys.argv[1])
    else:
        df_sample = process_data_sample("DataSample.csv")
    
    # Get POI Data
    if sys.argv[2]:
        df_poi = process_poi_list(filepath=sys.argv[2])
    else:
        df_poi = process_poi_list("POIList.csv")
    # Join the data
    df_joined = join_and_analyze(df_poi,df_sample)
    # Remove the outliers
    df_joined = remove_outliers(df_joined,threshold=3)
    # Extract Mean and Standard Deviation
    df_summary1 = calculate_mean_stdv(df_joined)
    # Extract Radius and Density
    df_summary2 = calculate_radius_and_density(df_joined)
    # Extend the POI data to include Radius information
    df_poi = df_poi.join(df_summary2,on="POIID",how="inner").toPandas()
    # Build the time metrics
    df_requests = build_time_metric(df_joined.toPandas(),dictionary=time_score)
    # Visualize the data
    if visualize:
        visualize_requests(df_requests,df_poi)
    # Run the model 
    df_results = run_model(df_requests)
    
    # Display the results
    print("Mean and Standard Deviation of the POIs\n")
    print(df_summary1)
    print("\nRadius and Density( Requests/Area) of the POIs\n")
    print(df_summary2.toPandas())
    print("\nRelative Popularity of Clusters\n")
    print(df_results)


# In[ ]:


if __name__ == "__main__":
    main(visualize=True)



# In[ ]:




