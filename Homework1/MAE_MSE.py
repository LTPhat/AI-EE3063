# Compare MAE and MSE

## MAE

#MAE (Mean absolute error) represents the difference between the original and predicted values extracted by averaged the absolute difference over the data set.

#$\text{MAE} = \frac{1}{N}\sum^{N}_{i = 1} \left|y_i- \hat{y}\right|$


## MSE

#MSE (Mean Squared Error) represents the difference between the original and predicted values extracted by squared the average difference over the data set.


#$\text{MSE} = \frac{1}{N}\sum^{N}_{i = 1} \left(y_i- \hat{y}\right)^2$

# Example
import numpy as np



def mae(y, y_hat):
    return np.mean(abs(y-y_hat))

def mse(y, y_hat):
    return np.mean(np.square(y-y_hat))

if __name__ == "__main__":

    y = np.array([-3, -1, -2, 1, -1, 1, 2, 1, 3, 4, 3, 5])
    yhat = np.array([-2, 1, -1, 0, -1, 1, 2, 2, 3, 3, 3, 5])
    mae_f = mae(y, yhat)
    mse_f = mse(y, yhat)
    print("MAE:",mae_f)
    print("MSE:", mse_f)