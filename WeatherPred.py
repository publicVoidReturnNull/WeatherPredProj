import pandas as pd
import networkClass as nn
import random

#Return a dataset that contains 2 columns, date/time and the corresponding temp, *ADDED WIND SPEED
#Additionally, reduce the size by a factor of 24 -> OG csv file goes by the hour
#Go by days, meaning -> combined every 24 indices into 1, find avg and use that for the day
def processData():
    data = pd.read_csv("/Users/joshuawang/Downloads/MLProj/MYNN/WeatherPred/Weather_Data.csv", usecols=['Date/Time', 'Temp_C', 'Wind Speed_km/h'])
    dataByDays = []
    dailyAvgTemp = 0
    dailyAvgWindSpd = 0
    count = 1
    for i in range(len(data)):
        dailyAvgTemp += data['Temp_C'][i]
        dailyAvgWindSpd += data['Wind Speed_km/h'][i]

        if i % 24 == 0:
            #creates dataSet that only contains 3 columns, date and the avgtemp and windspd throughout that day
            dataByDays.append((count-1, nn.Value(round((dailyAvgTemp / 24), 2)), nn.Value(round((dailyAvgWindSpd / 24), 2))))
            dailyAvgTemp = 0
            dailyAvgWindSpd = 0
            count+=1

    return dataByDays

def calcLoss(results, labels):
    # print(results)
    # print(labels)
    #for temp first, then wind spd
    l = sum((results[i][0] - labels[i][0])**2 for i in range(len(labels)))
    l += sum((results[i][1] - labels[i][1])**2 for i in range(len(labels)))
    return l

#Creates a neural network that takes in data (the date/time) and uses the temp and windspd as a label
#Data -> can change the list to store the avg temp relative to january 1st
#Therefore, can have a user input a date and the avg temp on january 1st of that year, 
# -> Have NN predict the temp and wind speed of the date inputted
def main():
    #data is a list of tuples -> (int, Value, Value)
    data = processData()

    #initializes the mlp from the MLP class in networkClass file (32 and 32 in hidden layer, outputs a temp and wind spd)
   
    mlp = nn.MLP(4, [32, 32, 2])
    labels = [(data[i][1], data[i][2]) for i in range(len(data))]
    inputs = [data[i][0] for i in range(len(data))]

    #change the input of the mlp to size of 4: the day you want to predict,
    #the date of the first day of the year(in this case: 1), the temp
    #of the day and the windspd of the day
    firstDayInfo = [data[0][0], labels[0][0], labels[0][1]]

    #how many iterations and how how much the nn learns by, and how many data points are used per epoch
    epochs = 25
    learning_rate = 0.0000001
    batch_size = 32
    batch_labels = []

    for epoch in range(epochs):
        #resets gradients and loss for new iteration 
        mlp.zero_grad()
        loss = 0
        results = []
        batch_inputs = random.sample(range(len(inputs)), batch_size)
        batch_labels = [labels[i] for i in batch_inputs]

        for inp in batch_inputs:
            #add inp to firstdayinfo, and then call mlp on firstDayInfo list
            results.append(mlp([inp] + firstDayInfo))

        loss = calcLoss(results, batch_labels)

        loss.backward()

        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
        # print(loss)
    
    # print(results)
    # print(batch_labels)

    #after iterations, try implement save the params in database or file, then take in user input, and predict 
    

    #ASSUMING USER GIVES AN INT/FLOAT
    firstDayTemp = (float(input("Enter first day of the year's temperature to 1 decimal: ")))
    firstDayWindSpd = (float(input("Enter first day of the year's wind speed to 1 decimal: ")))
    
    PredictedDay = int(input("Enter what day of the year's you want to predict: "))

    userInp = [PredictedDay, 1, firstDayTemp, firstDayWindSpd]

    pred = mlp([inp] + firstDayInfo)

    print(f"Temp of the {PredictedDay}th day of the year is {pred[0].data}")
    print(f"WindSpd of the {PredictedDay}th day of the year is {pred[1].data}")
    

main()
