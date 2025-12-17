from src.load_data import *
from src.evaluate import *
from src.model_1.train import train_model,predict
from src.model_2.train import train_model, predict_model


def main():
    print("Loading dataset...")
    train_ds,test_ds=load_data()
    X_train,y_train=preprocess_text(train_ds)
    X_test,y_test=preprocess_text(test_ds)

    print("Training Model 1...")
    model_1=train_model(X_train,y_train)

    print("Evaluating Model 1...")
    preds_1=predict(model_1,X_test)

    evaluate_model_1=evaluate(y_test,preds_1,model_name="Model 1")


    print("\n Training Model 2...")
    model_2=train_model(X_train,y_train)

    print("Evaluating Model 2...")
    preds_2=predict_model(model_2,X_test)

    evaluate_model_2=evaluate(y_test,preds_2,model_name="Model 2")



    print("modelcomparision")
    print("model_1:",evaluate_model_1)
    print("model_2:",evaluate_model_2)


    while True:
        user_input=input("Enter text to classify (or 'exit' to quit): ")
        if user_input.lower()=='exit':
            break
        pred_1=predict(model_1,[user_input])[0]
        pred_2=predict_model(model_2,[user_input])[0]
        print(f"Model 1 Prediction: {pred_1}")
        print(f"Model 2 Prediction: {pred_2}")

if __name__=="__main__":
    main()