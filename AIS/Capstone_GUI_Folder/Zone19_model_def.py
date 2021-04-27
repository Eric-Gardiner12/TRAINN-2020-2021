import numpy as np
import tensorflow as tf
from datetime import datetime

def Z19_MlModel(file_path1, file_path2, file_path3):

    #Loading Test data
    b = np.load(file_path1)
    mmsi_list = np.load(file_path2)
    type_check = np.load(file_path3,allow_pickle =True)
    

    #TFLITE
    interpreter = tf.lite.Interpreter(model_path = '/home/pi/Documents/Zone19_model.tflite')
    interpreter.allocate_tensors()

    #Get input/output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #Empty lists for adding the results to
    pie_data = []
    results = []

    #Testing model
    input_shape = input_details[0]['shape']
    for i in range(b.shape[0]):
        a = b[i]
        #print(a.shape)
        a = np.reshape(a, (1,73,4))
        #print(a.shape)
        #print(a.dtype)
        a = np.float32(a)
        #print(a.dtype)
        interpreter.set_tensor(input_details[0]['index'], a)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        #append results
        pie_data.append(output_data[0])
        results.append(np.argmax(output_data))


    #checking for suspicious activity
    sus_mmsi = []
    sus_mmsi_text = []
    
    #get today's date
    today = datetime.today().strftime('%Y-%m-%d')

    for j in range(len(results)):
        #check if predicted type doesn't match actual type
        if type_check[j] != results[j] and type_check[j] != None:
            if pie_data[j][results[j]] > 0.95:
                sus_mmsi.append(1)
                sus_mmsi_text.append(mmsi_list[j])
            else:
                sus_mmsi.append(None)
        else:
            sus_mmsi.append(None)
    
    if all(x is None for x in sus_mmsi):
        pass
    else:
        with open('/home/pi/Documents/{}_sus_mmsi.txt'.format(today),'w') as h:
            for element in sus_mmsi_text:
                h.write('%s\n' % element)
        
            
    return mmsi_list,pie_data,sus_mmsi, results
