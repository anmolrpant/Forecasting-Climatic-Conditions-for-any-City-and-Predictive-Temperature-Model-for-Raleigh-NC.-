#!/usr/bin/env python
# coding: utf-8

# ## Importing Python Libraries & Modules 

# In[1]:


#Importing Python Libraries for the Project

#Python Libraries for Current Weather Conditions
import requests
from bs4 import BeautifulSoup
from ISStreamer.Streamer import Streamer
import urllib.request as urllib2
import json
import webbrowser

#Python Libraries for Predictive Average Temperature Model
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Visualizing and Streaming the Current Weather Conditions of a City

# In[2]:



#Function to Extract Weather Conditions from Dark Sky API into a Json object and then returning the results to main function
def get_current_conditions():
    api_conditions_url = "https://api.darksky.net/forecast/"+DARKSKY_API_KEY+"/"+GPS_COORDS+"?units=auto"
    try:
        weather_data = urllib2.urlopen(api_conditions_url)    #opening the weather API
    except:
        return []
    json_currently = weather_data.read()                     #reading data into Json object
    weather_data.close()
    return json.loads(json_currently)



#Function to check if the values extracted are decimal numbers or not and return boolean result
def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
    
    
#Function to Determine the Moon Phase Icon based on Moon Phase value    
def moon_icon(moon_phase):
    if moon_phase < .06:
        return ":new_moon:"
    if moon_phase < .125:
        return ":waxing_crescent_moon:"
    if moon_phase < .25:
        return ":first_quarter_moon:"
    if moon_phase < .48:
        return ":waxing_gibbous_moon:"
    if moon_phase < .52:
        return ":full_moon:"
    if moon_phase < .625:
        return ":waning_gibbous_moon:"
    if moon_phase < .75:
        return ":last_quarter_moon:"
    if moon_phase < 1:
        return ":waning_crescent_moon:"
    return ":crescent_moon:"



#Function to Determine the Weather Condition Icon based on Moon Phase value 
def weather_icon(ds_icon):
    icon = {
        "clear-day"             : ":sunny:",
        "clear-night"           : ":new_moon_with_face:",
        "rain"                  : ":umbrella:",
        "snow"                  : ":snowflake:",
        "sleet"                 : ":sweat_drops: :snowflake:",
        "wind"                  : ":wind_blowing_face:",
        "fog"                   : ":fog:",
        "cloudy"                : ":cloud:",
        "partly-cloudy-day"     : ":partly_sunny:",
        "partly-cloudy-night"   : ":new_moon_with_face:",
        "unknown"               : ":sun_with_face:",
    }
    return icon.get(ds_icon,":sun_with_face:")


#Function Calling to get Moon Phase based on Weather Icon
def weather_status_icon(ds_icon,moon_phase):
    icon = weather_icon(ds_icon)
    if (icon == ":new_moon_with_face:"):
        return moon_icon(moon_phase)
    return icon



#Function to Determine the Wind Direction Icon based on Moon Phase valueWind Bearing Value from Weather API
def wind_dir_icon(wind_bearing):
    if (wind_bearing < 20):
        return ":arrow_up:"
    if (wind_bearing < 70):
        return ":arrow_upper_right:"
    if (wind_bearing < 110):
        return ":arrow_right:"
    if (wind_bearing < 160):
        return ":arrow_lower_right:"
    if (wind_bearing < 200):
        return ":arrow_down:"
    if (wind_bearing < 250):
        return ":arrow_lower_left:"
    if (wind_bearing < 290):
        return ":arrow_left:"
    if (wind_bearing < 340):
        return ":arrow_upper_left:"
    return ":arrow_up:"



def main():
    city = input("Enter the name of the City you require the Weather Conditions of:")
    address = "https://www.google.com/search?rlz=1C1HLDY_enUS874US874&ei=dengXeHcE8qd5wKj4aDIAw&q="+city+"+coordinates&oq="+city+"+&gs_l=psy-ab.3.0.0i67i70i251j0j0i67l3j0i131j0i67l3j0i131.10435.11689..12656...0.1..0.165.881.4j4......0....1..gws-wiz.......0i71j0i273.KfscqkpHNeY"
    web_page = requests.get(address)
    soup = BeautifulSoup(web_page.content, "html.parser")                #Scraping Content from Google Search
    coords_class = soup.find_all(class_="BNeawe iBp4i AP7Wnd")
    required_coords =coords_class[0].find_all(class_="BNeawe iBp4i AP7Wnd")    #Getting coords in a particular class in scraped data
    latitudes_longitude=[]
    str_coords= str(required_coords)
    t=str_coords.replace(">"," ")
    t=t.replace("<"," ")
    t=t.replace(","," ")
    t=t.replace("°","")
    T=t.split(" ")
    latitudes_longitude.append(T[5])
    latitudes_longitude.append(T[6])
    latitudes_longitude.append(T[8])
    latitudes_longitude.append(T[9])
    coords =""
    if(latitudes_longitude[1]=="S"):
        latitudes_longitude[0]=str(-1.0*float(latitudes_longitude[0]))   #Converting Latitude to standard notation
    if(latitudes_longitude[3]=="W"):
        latitudes_longitude[2]=str(-1.0*float(latitudes_longitude[2]))   #converting Longitude to standar notation
    coords = latitudes_longitude[0]+","+latitudes_longitude[2]           #Adding Latitudes and Longitudes of a place in "coords" variable
    coords
    global GPS_COORDS  
    GPS_COORDS = coords
    global DARKSKY_API_KEY
    
    DARKSKY_API_KEY = "c0974690b3b66a08fe5ea57c4926f420"            #Storing API key in DarkSky_API_KEY variable
    BUCKET_NAME = ":partly_sunny: "+city+" Weather"                 #Storing Bucket Name
    BUCKET_KEY = "T57BL7K39XAX"                                     #Storing Buckey Key for Streamer Dashboard 
    ACCESS_KEY = "ist_uOnFF2jMhg7iwAu9R5HqFVAVeTZoQk-q"             #Storing Access Key of Streamer Dashboard 
    
    
    curr_conditions = get_current_conditions()
    if ('currently' not in curr_conditions):      # Checking for connectivity to API and printing if connection is invalid
        print ("Error! Dark Sky API call failed, check your GPS coordinates and make sure your Dark Sky API key is valid!\n")
        print (curr_conditions)
        exit()
    else:
        streamer = Streamer(bucket_name=BUCKET_NAME, bucket_key=BUCKET_KEY, access_key=ACCESS_KEY)  #Accessing the Streamer
    while True:
        curr_conditions = get_current_conditions()
        if ('currently' not in curr_conditions):
            print ("Error! Dark Sky API call failed. Skipping a reading then continuing ...\n")  #Again checking connectivity
        else:
            streamer.log(":house: Location",GPS_COORDS)        #Streaming the Location on the Dashboard
            webbrowser.open('https://go.init.st/jp3ahqe')      #Command to automatically open up default webbrowser
            
            #Streaming the Humidity on the Dashboard
            if 'humidity' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['humidity']):
                streamer.log(":droplet: Humidity(%)", curr_conditions['currently']['humidity']*100)
                
            #Streaming the Temperature on the Dashboard 
            if 'temperature' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['temperature']): 
                streamer.log("Temperature",curr_conditions['currently']['temperature'])

            #Streaming the Apparent Temperature on the Dashboard     
            if 'apparentTemperature' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['apparentTemperature']): 
                streamer.log("Apparent Temperature",curr_conditions['currently']['apparentTemperature'])
            
            #Streaming the DewPoint on the Dashboard 
            if 'dewPoint' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['dewPoint']):
                streamer.log("Dewpoint",curr_conditions['currently']['dewPoint'])

            #Streaming the WindSpeed on the Dashboard 
            if 'windSpeed' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['windSpeed']):
                streamer.log(":dash: Wind Speed",curr_conditions['currently']['windSpeed'])

            #Streaming the Wind_Gust on the Dashboard 
            if 'windGust' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['windGust']):
                streamer.log(":dash: Wind Gust",curr_conditions['currently']['windGust'])
                
            #Streaming the Wind Direction on the Dashboard 
            if 'windBearing' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['windBearing']):
                streamer.log(":dash: Wind Direction",wind_dir_icon(curr_conditions['currently']['windBearing']))
            
            #Streaming the Pressure Condition on the Dashboard 
            if 'pressure' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['pressure']):
                streamer.log("Pressure",curr_conditions['currently']['pressure'])
                
            #Streaming the Precipitation Intensity on the Dashboard 
            if 'precipIntensity' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['precipIntensity']):
                streamer.log(":umbrella: Precipitation Intensity",curr_conditions['currently']['precipIntensity'])
              
            #Streaming the Precipitation Probability on the Dashboard
            if 'precipProbability' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['precipProbability']):
                streamer.log(":umbrella: Precipitation Probabiity(%)",curr_conditions['currently']['precipProbability']*100)
                
            #Streaming the Cloud Cover on the Dashboard
            if 'cloudCover' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['cloudCover']):
                streamer.log(":cloud: Cloud Cover(%)",curr_conditions['currently']['cloudCover']*100)
                
            #Streaming the Ultraviolet Index on the Dashboard
            if 'uvIndex' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['uvIndex']):
                streamer.log(":sunny: UV Index:",curr_conditions['currently']['uvIndex'])
                
            #Streaming the Visibility on the Dashboard
            if 'visibility' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['visibility']):
                streamer.log(":sunny: Visibility:",curr_conditions['currently']['visibility'])
            
            #Streaming the Ozone Level on the Dashboard
            if 'ozone' in curr_conditions['currently'] and isFloat(curr_conditions['currently']['ozone']):
                streamer.log(":sunny: Ozone Level:",curr_conditions['currently']['ozone'])
                
            #Streaming the Weather Summary on the Dashboard
            if 'summary' in curr_conditions['currently']:
                streamer.log(":cloud: Weather Summary",curr_conditions['currently']['summary'])
            
            #Streaming the Forecast Message for today on the Dashboard
            if 'hourly' in curr_conditions:
                streamer.log("Today's Forecast",curr_conditions['hourly']['summary'])
                
            #Streaming the Moon Phase and Weather Condition Icon on the Dashboard
            if 'daily' in curr_conditions:
                if 'data' in curr_conditions['daily']:
                    if 'moonPhase' in curr_conditions['daily']['data'][0]:
                        moon_phase = curr_conditions['daily']['data'][0]['moonPhase']
                        streamer.log(":crescent_moon: Moon Phase",moon_icon(moon_phase))
                        streamer.log(":cloud: Weather Conditions",weather_status_icon(curr_conditions['currently']['icon'],moon_phase))
        
            streamer.flush()         #Refreshing the Stream
            streamer.close()         #Closing the Stream
        break   
    


# ## Predictive  Average Temperature Regression Model for city of Raleigh, North Carolina

# In[6]:


def regress():
    Weather = pd.read_excel("Weather Conditions for Raleigh City from 1990-2019.xlsx")    # reading a dataframe using Pandas
    Weather = Weather.replace('M',0)

    #Converting object datatype into Numeric Datatype
    Weather['Average Temperature'] = pd.to_numeric(Weather['Average Temperature'])
    Weather['Max Temperature'] = pd.to_numeric(Weather['Max Temperature'])
    Weather['Min Temperature'] = pd.to_numeric(Weather['Min Temperature'])
    Weather['Mean Precipitation'] = pd.to_numeric(Weather['Mean Precipitation'])
    Weather['Mean Snowfall'] = pd.to_numeric(Weather['Mean Snowfall'])
    Weather['Snow Depth'] = pd.to_numeric(Weather['Snow Depth'])
    Weather['Growing Degree Day'] = pd.to_numeric(Weather['Growing Degree Day'])
    Weather['Heating Degree Day'] = pd.to_numeric(Weather['Heating Degree Day'])
    Weather['Cooling Degree Day'] = pd.to_numeric(Weather['Cooling Degree Day'])
    print("\n\nThe Weather dataset head is:\n" ,Weather.head())
    
    #Merging two Columns of Month & Year into one column and dropping previous ones
    Weather['MY'] = Weather[Weather.columns[:-10]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    idx = 0
    new_col = Weather["MY"]  # can be a list, a Series, an array or a scalar   
    Weather.insert(loc=idx, column="Month & Year", value=new_col)
    Weather = Weather.drop(["Year", "Month","MY"],axis=1)
    
    # Storing Values of Response and Predictors in y and X respectively
    X = Weather[["Max Temperature","Min Temperature","Mean Precipitation","Mean Snowfall","Snow Depth","Heating Degree Day","Cooling Degree Day","Growing Degree Day"]].values
    y = Weather['Average Temperature'].values
    
    #Assigning values to training and test values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
    
    
    #Model 1 using Simple Linear Regression
    regressor = LinearRegression()                       #Linear Regression
    fit = regressor.fit(X_train, y_train)                #Regreesion Fit
    coeff_df = pd.DataFrame(regressor.coef_, index = Weather[["Max Temperature","Min Temperature","Mean Precipitation","Mean Snowfall","Snow Depth","Heating Degree Day","Cooling Degree Day","Growing Degree Day"]].columns,columns=["Coefficient"])
    y_pred = regressor.predict(X_test)                   #Predicting the test dataset
    r2=metrics.r2_score(y_test,y_pred)                   #Finding the Accuracy of the Model
    print("\nAccuracy of Simple Linear Regression Model is:" ,r2)

    #Making a dataframe of Actual and Predicted value to plot the results and show the Error in our model
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df1 = df.head(25)
    print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    
    
    #Model 2 using Ordinary Least Square Method -
    #OLS is available in the statsmodel.api library
    model = sm.OLS(y_train,X_train).fit()                 #Ordinary Least Square Regression
    predictions = model.predict(X_test)                   #Prediction of Test dataset on OLS
    print("The summary of the OLS model is \n \n \n:" ,model.summary())      #printing the summary of OLS model
    

    #Visualization of Data Plots for the Average Temperature over Time -
    plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.barplot(x = Weather["Month & Year"],y = Weather['Average Temperature'])
    ax.set_title('Average Temperature Movement from (1990-2019)')
    ax.set_ylabel('Average Temperature for each Month')
    ax.set_xlabel('Date (1990 - 2019)')
    
    #Visualization of Variation of Temperature Intervals
    plt.figure(figsize=(16,6))
    plt.tight_layout()
    sns.distplot(Weather['Average Temperature'])
    plt.title("Average Temperature Seasonality")
    plt.show()
    
    
    #Visualization of Actual vs Predicted Value for Simple Linear Regression
    df1.plot(kind='bar',figsize=(12,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title("Bar Graph for Actual vs Predicted Values for Simple Linear Regression")
    plt.show()
    
    #Visualization of Actual vs Predicted Value for OLS
    df2 = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
    df2.plot(kind='line',figsize=(12,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title("Line Graph for Actual vs Predicted Values for Ordinary Least Squares")
    plt.show()
    
    
    #Forecast the Average Temperature of the end of the Day based on User Values for different current weather components
    print("User_Input:")
    print("Enter weather details to predict average Temperature of Raleigh")
    User=[]
    User.append(float(input("Maximum Temperature:")))
    User.append(float(input("Minimum Temperature:")))
    User.append(float(input("Precipitation:")))
    User.append(float(input("Snowfall:")))
    User.append(float(input("Snow Depth:")))
    User.append(float(input("HDD:")))
    User.append(float(input("CDD:")))
    User.append(float(input("GDD:")))
    user_value = pd.DataFrame([list(User)],columns=Weather[["Max Temperature","Min Temperature","Mean Precipitation","Mean Snowfall","Snow Depth","Heating Degree Day","Cooling Degree Day","Growing Degree Day"]].columns)
    val =user_value.values
    new_pred = regressor.predict(val)
    print("The Forecasted Average Temperature by the end of the day for Tomorrow is", new_pred)


# ## Main Function to Select our Choice

# In[14]:


if __name__ == "__main__":
    print("Choice 1: Current Weather Conditions of a City in the World")
    print("Choice 2: Predicitve Analytical Model for Average Temperature for the City of Raleigh, North Carolina")
    
    choice = input("Enter your choice:")        #User_Choice 
    if(choice == "1"):
        main()
    if(choice =="2"):
        regress()
    else:
        print("Wrong Choice")


# In[ ]:

# # THANK YOU

# In[ ]:




