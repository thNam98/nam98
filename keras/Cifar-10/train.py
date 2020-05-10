from architecture import MPL, get_data
if __name__ == "__main__":
    # parameter
    epochs = 5
    batch_size = 128
    model_path = "model/mnist_keras.h5"

    # process
    # get_data
    x_train, y_train, x_test, y_test = get_data.get_data_reshape()
    print("Number of train data:", len(x_train))
    print("Number of test data:", len(x_test))

    # get model
    MLP_Mnist_model = MPL.get_model()

    # train
    MLP_Mnist_model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=True,
                        validation_data=(x_test, y_test))

    # evaluate
    score = MLP_Mnist_model.evaluate(x_test, y_test, verbose=False)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Score F1:', MPL.f1)

    # save model
    MLP_Mnist_model.save(model_path)
