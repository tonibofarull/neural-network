from nn import NN
import numpy as np
import matplotlib.pyplot as plt
import pickle

DRAW_PLOT = False
NAME_FILE = "spiral"

def get_dots(name="array"):
    plt.clf()
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    print("Draw one class")
    red = list(zip(*plt.ginput(n=-1, show_clicks=True)))
    plt.scatter(red[0], red[1])
    print("Draw the other class")
    green = list(zip(*plt.ginput(n=-1, show_clicks=True)))

    xy = [[],[]]
    xy[0] = red[0] + green[0]
    xy[1] = red[1] + green[1]

    X = np.array([xy[0], xy[1]])

    reds = len(red[0])
    greens = len(green[0])

    Y = np.array([0]*reds + [1]*greens).reshape((1,-1))
    pickle.dump((X,Y,reds,greens), open("../data/" + name,"wb"))

def get_train(new_plot=True, name="array"):
    if new_plot:
        get_dots(name)
    W = pickle.load(open("../data/" + name,"rb"))
    X, Y, reds, greens = W[0], W[1], W[2], W[3]

    print("Problem to classify")
    print("Close figure to continue...")
    plt.scatter(X[0], X[1], c=[[1,0,0]]*reds + [[0,1,0]]*greens)
    plt.show()
    return X, Y, reds, greens

def get_test():
    Xtest = []
    for x in np.linspace(-1,1,100):
        for y in np.linspace(-1,1,100):
            Xtest.append([x,y])
    Xtest = np.asarray(Xtest).T
    return Xtest

def get_contour(A, Y):
    res = []
    for x in A.T:
        x = x[0]
        res.append([x,x,x])
    return res

if __name__ == "__main__":

    nn = NN([2,8,8,1])

    X, Y, reds, greens = get_train(new_plot=DRAW_PLOT, name=NAME_FILE)
    Xtest = get_test()

    fig = plt.figure()
    fig.show()
    for i in range(30):

        nn.train(X, Y, iters=100, t=i)
        A = nn.predict(Xtest)
        res = get_contour(A, Y)

        plt.clf()
        plt.scatter(Xtest[0], Xtest[1], c=res)
        plt.scatter(X[0], X[1], c=[[1,0,0]]*reds + [[0,1,0]]*greens)

        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        plt.text(-0.95,0.95, f"epoch {(i+1)*100}\ncost = {round(nn.costs[-1],8)}", 
            fontsize=8,verticalalignment='top', bbox=props)
        fig.canvas.draw()

    plt.show()
    plt.plot(nn.costs)
    plt.title("Cost Function")
    plt.xlabel("epoch")
    plt.show()