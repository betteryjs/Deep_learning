from tensorflow.keras.preprocessing import image

# from tensorflow.python.keras.optimizers import SGD

def main():
    ml=image.load_img("C:/images/ML.png")
    ml=image.img_to_array(ml)
    print(ml)



if __name__ == '__main__':
    main()
