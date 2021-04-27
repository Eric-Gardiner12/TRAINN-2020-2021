#Data preparation for Daily ship CSV Files
import pandas as pd
import numpy as np

def Data_prep(file_path):

    #Read in data
    cols = ['Index',' Date', ' Time', ' mmsi', ' Lat', ' Long', ' Sog', ' Cog', ' VesselName']
    train_df = pd.read_csv(file_path, usecols = cols)

    #fix column names
    train_df = train_df.rename(columns = {'Index':'Date',' Time': 'MMSI',' Date':'Time',' mmsi':'LAT', ' Lat': 'LON', ' Long':'SOG',' Sog': 'COG',' Cog': 'VesselType' ,' VesselName': 'Status' })
    
    #Normalize values
    max_lat = 89.96446
    max_lon = -66.00001
    max_sog = 51.1
    max_cog = 204.7
    min_lat = 0.0001
    min_lon = -72.0
    min_sog = -51.20000076293945
    min_cog = -204.8000030517578

    train_df['LAT'] = (train_df['LAT']- min_lat) / (max_lat - min_lat)
    train_df['LON'] = (train_df['LON']- min_lon) / (max_lon - min_lon)
    train_df['SOG'] = (train_df['SOG']- min_sog) / (max_sog - min_sog)
    train_df['COG'] = (train_df['COG']- min_cog) / (max_cog - min_cog)
    
    #keep only required rows, those with enough time steps
    train_df = train_df.dropna(subset=['MMSI', 'Time', 'LAT', 'SOG', 'LON', 'COG', 'Date'])
    train_df = train_df.sort_values(by=['MMSI', 'Time'])
    #train_df = train_df[train_df.groupby('MMSI').MMSI.transform(len) > 99]
    train_df = train_df.reset_index(drop = True)

    #remove whitespaces
    train_df['MMSI'] = train_df['MMSI'].str.lstrip()
    train_df['Time'] = train_df['Time'].str.lstrip()
    train_df['VesselType'] = train_df['VesselType'].str.lstrip()
    train_df['Status'] = train_df['Status'].str.lstrip()
    train_df['Date'] = train_df['Date'].str.lstrip()

    #Add PosTime column for later resampling/interpolating
    train_df["PosTime"] = train_df["Date"] + ' ' +train_df["Time"]
    train_df['PosTime'] = pd.to_datetime(train_df['PosTime'])
    unique_date = train_df.Date[0]
    train_df = train_df.drop(['Date'], axis = 1)
    train_df = train_df.drop(['Time'], axis = 1)

    #remove those at anchor and moored
    index_names2 = train_df[ train_df['Status'] == 'NavigationStatus.AtAnchor' ].index 
    train_df.drop(index_names2, inplace = True)
    index_names3 = train_df[ train_df['Status'] == 'NavigationStatus.Moored' ].index
    train_df.drop(index_names3, inplace = True)
    train_df = train_df.drop('Status', axis = 1)

    unique_MMSIs = train_df.MMSI.unique()
    #Loop through all ships' time series
    final_input_train = np.empty([1,73,4])
    unique_MMSI = []
    unique_output = []
    for j in range(len(unique_MMSIs)):
        try:
            train_input = train_df[train_df.MMSI == unique_MMSIs[j]]
            train_input.reset_index(inplace = True)
            unique_id = train_input.MMSI[0]
            unique_types = train_input.VesselType.unique()
            cleanedList = [x for x in unique_types if str(x) != 'nan']
            train_input = train_input.set_index('PosTime')
            train_input = train_input.drop('MMSI', axis = 1)
            train_input = train_input.drop('VesselType', axis = 1)
            norm_train_df = pd.DataFrame()
            norm_train_df['Lat'] = train_input.LAT.resample('10T').last()
            norm_train_df['Lon'] = train_input.LON.resample('10T').last()
            norm_train_df['Sog'] = train_input.SOG.resample('10T').last()
            norm_train_df['Cog'] = train_input.COG.resample('10T').last()
            norm_train_df['Lat'] = pd.to_numeric(norm_train_df['Lat'], errors='coerce')
            norm_train_df['Lon'] = pd.to_numeric(norm_train_df['Lon'], errors='coerce')
            norm_train_df['Sog'] = pd.to_numeric(norm_train_df['Sog'], errors='coerce')
            norm_train_df['Cog'] = pd.to_numeric(norm_train_df['Cog'], errors='coerce')
            norm_train_df = norm_train_df.interpolate(method='spline', order=3, s=0.)
            norm_train_df.reset_index(inplace = True)
            norm_train_df = norm_train_df.iloc[0:73]
            norm_train_df = norm_train_df.drop('PosTime', axis = 1)
            norm_train_df = norm_train_df.values
            norm_train_df = np.reshape(norm_train_df, (1,73,4))
            final_input_train = np.append(final_input_train, norm_train_df, axis = 0)
            unique_MMSI.append(unique_id)
            if len(cleanedList) != 0:
                unique_output.append(cleanedList[0])
            else:
                unique_output.append(None)
        except:
            pass

    #convert dataframe to numpy array
    #train_df = np.array(list(train_df.groupby('MMSI').apply(pd.DataFrame.to_numpy)))
    #remove dummy variable at the beginning
    final_input_train= np.delete(final_input_train,0,0)

    #Check if the ships had vessel codes
    check = True
    if all(x is None for x in unique_output):
        check = False
    else:
        check = True
    if check is True:
        subs = {'ShipType.Fishing':1,'ShipType.Cargo':0,'ShipType.NotAvailable':None,'ShipType.SearchAndRescueVessel':3,'ShipType.Passenger':4,'ShipType.PleasureCraft':5,'ShipType.OtherType':3,'ShipType.SPARE':3,'ShipType.Tanker':6,'ShipType.OtherType_NoAdditionalInformation':3,'ShipType.DivingOps':3,'ShipType.Tug':2,'ShipType.Tow':2}
        C = (pd.Series(unique_output)).map(subs) #convert the list to a pandas series temporarily before mapping
        D = list(C) # we transform the mapped values (a series object) back to a list
        for k in range(len(D)):
            if str(D[k]) != 'nan':
                D[k] = int(D[k])
    
    #store inputs/MMSIs as numpy arrays
    with open('/home/pi/Documents/{}_inputs.npy'.format(unique_date),'wb') as f:
        np.save(f,final_input_train)
    with open('/home/pi/Documents/{}_MMSIs.npy'.format(unique_date),'wb') as g:
        np.save(g,unique_MMSI)
    
    #store outputs as numpy array if they were given
    #if check != False:
    with open('/home/pi/Documents/{}_outputs.npy'.format(unique_date),'wb') as h:
        np.save(h,unique_output)
    return check,final_input_train.shape[0]

    
    

