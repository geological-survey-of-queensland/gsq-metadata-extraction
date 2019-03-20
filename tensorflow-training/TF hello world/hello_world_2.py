import tensorflow as tf
import numpy as np

def main():

    x_train = arrayFromFile("TrainDigitX.idx")
    y_train = arrayFromFile("TrainDigitY.idx")
    x_test = arrayFromFile("TestDigitX.idx")
    y_test = arrayFromFile("TestDigitY.idx")

    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(784,)),
    tf.keras.layers.Dense(100, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=2)
    print(model.evaluate(x_test, y_test))

# read file into a np array
def arrayFromFile(filename):

    print("reading", filename)

    data = bytes()
    dimensions = []
    size = 0
    data = data
    dataTypeSize = 1
    startingIndex = 0

    with open(filename,'rb') as file:
        string = file.read()
        data = bytes(string)

    dataType = data[2]

    if dataType == 0x8 or dataType == 0x9:
        dataTypeSize = 1
    elif dataType == 0xB:
        dataTypeSize = 2
    elif dataType == 0xC or dataType == 0xD:
        dataTypeSize = 4
    elif dataType == 0xE:
        dataTypeSize = 8

    dimensionCount = data[3]
    index = 4

    for d in range(dimensionCount):

        size = 0

        for i in range(4):
            size = size<<8 | data[index+i]

        dimensions.append(size)
        index += 4

    startingIndex = index
    dataList = list(data[startingIndex:])
    array = np.array(dataList)
    array = array.astype(float)

    if dimensionCount == 3:
        array = array.reshape(dimensions[0], len(dataList)//dimensions[0])
        array /= 255

    return array

# after all functions are defined, run main code
if __name__ == "__main__": main()