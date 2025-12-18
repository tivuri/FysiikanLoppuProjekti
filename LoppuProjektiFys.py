import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import folium
from streamlit_folium import st_folium
from math import radians, cos, sin, asin, sqrt
import streamlit as st

askelData = pd.read_csv('Linear Acceleration.csv')
locationData = pd.read_csv('Location.csv')

st.title("Käppäily reissu by Tino Vuori, TVT24KMO")

# Askeleet suodatettuna

def butter_lowpass_filter(askelData, cutOff, nyq, order):
    normal_cutOff = cutOff / nyq
    b, a = butter(order, normal_cutOff, btype='low', analog=False)
    y = filtfilt(b, a, askelData)
    return y

data = askelData['Linear Acceleration y (m/s^2)']
totalTime = askelData['Time (s)'].max()
dataPisteet = len(askelData['Time (s)'])
taajuus = dataPisteet/totalTime
nyq = taajuus / 2
order = 3
cutOff = 1 / 0.4
filteredData = butter_lowpass_filter(data, cutOff, nyq, order)

askeleet = 0

for i in range(dataPisteet-1):
    if filteredData[i] / filteredData[i+1] < 0:
        askeleet = askeleet + 1/2

st.write("Askelien määrä suodatuksesta: ", askeleet, " askelta")

# Askeleet laskettuna fourier-muunnoksen avulla

askeleetFourier = askelData['Linear Acceleration y (m/s^2)']
time = askelData['Time (s)']
dataPoints = len(askeleetFourier)
samplingInterval = np.max(time) / dataPoints

fourier = np.fft.fft(askeleetFourier, dataPoints)
spectralDensity = fourier*np.conj(fourier)/dataPoints
frequency = np.fft.fftfreq(dataPoints, samplingInterval)
cropped = np.arange(1, int(dataPoints/2))

frequencyMax = frequency[cropped][spectralDensity[cropped] == np.max(spectralDensity[cropped])][0]
stepTime = 1 / frequencyMax
calculatedSteps = frequencyMax*np.max(time)

st.write('Askelmäärä fourier-analyysin avulla: ', np.round(calculatedSteps), " askelta")

# kuljettu matka
def haversine(longitude1, latitude1, longitude2, latitude2):
    longitude1, latitude1, longitude2, latitude2 = map(radians, [longitude1, latitude1, longitude2, latitude2])

    distanceLongitude = longitude2 - longitude1
    distanceLatitude = latitude2 - latitude1
    a = sin(distanceLatitude/2)**2 + cos(latitude1) * cos(latitude2) * sin(distanceLongitude/2)**2
    c = 2 * asin(sqrt(a))
    radius = 6371
    return c * radius

locationData['DistanceCalculated'] = np.zeros(len(locationData))

for i in range(len(locationData)-1):
    longitude1 = locationData['Longitude (°)'][i]
    longitude2 = locationData['Longitude (°)'][i+1]
    latitude1 = locationData['Latitude (°)'][i]
    latitude2 = locationData['Latitude (°)'][i+1]
    locationData.loc[i+1, 'DistanceCalculated'] = haversine(longitude1, latitude1, longitude2, latitude2)

totalDistanceKm = locationData['DistanceCalculated'].sum()
st.write("Kävelty matka: ",totalDistanceKm.round(2), "km")

# Keskinopeus
# speed(m/s) = distance (m) / time (s)
averageSpeed = (totalDistanceKm * 1000) / totalTime
st.write("Keskinopeus: ", averageSpeed.round(2), "m/s")

# Suodatetun y-komponentin kuvaajat

st.title("Suodatettu data y-komponentin kahdesta aikavälistä, jotta näkee nopeuden eron")
fig, ax = plt.subplots(figsize=(16,6))
plt.plot(askelData['Time (s)'], filteredData, label = 'Suodatettu data')
plt.xlim(10, 70)
plt.xlabel('Time (s)')
plt.ylabel('Suodatettu y (m/s^2)')
plt.grid()
plt.legend()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(16,6))
plt.plot(askelData['Time (s)'], filteredData, label = 'Suodatettu data')
plt.xlim(410, 470)
plt.xlabel('Time (s)')
plt.ylabel('Suodatettu y (m/s^2)')
plt.grid()
plt.legend()
st.pyplot(fig)


#Tehospektri

st.title("Tehospektri")
fig, ax = plt.subplots(figsize=(16,6))
plt.plot(frequency[cropped], spectralDensity[cropped].real)
plt.title('Tehospektri')
plt.ylabel('Teho')
plt.xlabel('Taajuus [Hz] = [1/s]')
plt.xlim(0, 16)
plt.grid()
st.pyplot(fig)


#Karttakuva

latitude1 = locationData['Latitude (°)'].mean()
longitude1 = locationData['Longitude (°)'].mean()
st.title("Karttakuva")
myOwnMap = folium.Map(location = [latitude1, longitude1], zoom_start = 15)
folium.PolyLine(locationData[['Latitude (°)', 'Longitude (°)']], color = 'red', weight = 3).add_to(myOwnMap)
stMap = st_folium(myOwnMap, width = 600, height = 400)
myOwnMap.save('FysiikanLoppuProjektiKartta.html')





