import pickle
import numpy as np
def predict( Xarray):
    with open('rf_model.pickle', 'rb') as to_read:
        rf_cv = pickle.load(to_read)
    arr = np.array(Xarray).reshape(1,-1)
    print(rf_cv)
    return rf_cv.predict(arr)

# Example usage (Drop timestamps ,and activity columns):
print(predict([-1.008,0.229,-0.072,-3.537,-2.073,-0.305,0,-0.973,0.301,0.103,-5.366,-1.28,-0.732,0,0.16,0.895,0.372,2.012,4.634,3.354,392.633,-0.727,0.095,-0.758,4.878,2.866,-3.232,1231.336,-1.597,-0.646,0.112,10,30.671,126.768,2108.154,-436,1,1,1,1,1,1,1,1,7]))

