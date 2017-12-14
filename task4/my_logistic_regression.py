from numpy import append, genfromtxt, array, ndarray, e, exp, log
import pandas as pd

class MyLogisticRegression():

    def __init__(self, data, header=False):
        if type(data) == ndarray:
            if header:
                data = data[1:]
            else:
                data = data
            self.data = data
        elif type(data) == str:
            self.data = self.load_data(data, header)

    def score(self, w_array, array_points):
        total_score = 0
        for i in range(len(array_points)):
            tuple_len = len(array_points[i])
            x_array = append(array(1), array_points[i, 0:tuple_len - 1])

            total_score += self.sum_products(w_array, x_array)

        return total_score

    def sigmoid(self, score):
        # print("score", score)
        return 1 / (1 + (e ** (-score)))

    def sum_products(self, w_array, x_array):
        assert len(w_array) == len(x_array)

        total = 0
        for i in range(len(w_array)):
            total += w_array[i] * x_array[i]

        return total

    def step_gradient(self, current_w_array, array_points, learning_rate):
        # gradient_descent
        w_gradient_array = [0] * array_points.shape[1]

        for i in range(len(array_points)):
            tuple_len = len(array_points[i])

            x_array = append(array(1), array_points[i, 0:tuple_len - 1])
            y = array_points[i, tuple_len - 1]
            for j in range(len(w_gradient_array)):
                w_gradient_array[j] += self.gradient_ascent_calc(current_w_array, array_points, y, x_array[j])

        # update coefficients
        new_w_array = [None] * len(current_w_array)
        for i in range(len(current_w_array)):
            new_w_array[i] = current_w_array[i] - (2 * learning_rate * w_gradient_array[i])

        return new_w_array

    def likelihood(self, w_array, array_points):
        total = 1
        for i in range(len(array_points)):
            p_sigmoid = self.sigmoid(self.score(w_array, array_points))
            # print("p_sigmoid", p_sigmoid)
            # print("log(p_sigmoid)", log(p_sigmoid))
            total = total * log(p_sigmoid)

        return p_sigmoid

    def gradient_ascent_calc(self, current_w_array, array_points, y, x):
        return x * (self.indicator(y) - self.likelihood(w_array=current_w_array, array_points=array_points))

    def indicator(self, y):
        return y


    def gradient_ascent_runner(self, points, initial_w_array, learning_rate, num_iterations, cost_tolerance, verbosity=False):
        w_array = initial_w_array

        score = self.score(w_array, array_points=points)

        iterations_count = 0
        while iterations_count <= num_iterations:
            iterations_count += 1
            print("w_array", w_array)
            w_array = self.step_gradient(w_array, array(points), learning_rate)


            score = self.score(w_array, array_points=points)

            if verbosity == "vv":
                print("Current RSS:", score)  # item 2

        if verbosity == "v" or verbosity == "vv":
            print()
            print("---\nFinal RSS:", score)
        return w_array, score

    def predict(self, data):
        preds = []
        if type(data) == type(pd.DataFrame()):
            test = pd.DataFrame(data.as_matrix())

        for index, row in test.iterrows():
            values = row.tolist()
            pred = self.w_array[0]
            for i in range(len(self.w_array) - 1):
                pred += self.w_array[i + 1] * values[i]

            preds.append(pred)

        return preds

    def run(self, learning_rate=0.00001, num_iterations=5000, cost_tolerance=float("-inf"), verbosity=False):
        initial_w_array = [0] * self.data.shape[1]

        self.w_array, self.rss = self.gradient_ascent_runner(self.data, initial_w_array, learning_rate, num_iterations, cost_tolerance,
                                                             verbosity)

        if verbosity == "v" or verbosity == "vv":
            print("")
            for i in range(len(self.w_array)):
                print("w{}:".format(i), self.w_array[i])

        return self.w_array, self.rss

    def load_data(self, data_filename, header=False):
        points = genfromtxt(data_filename, delimiter=",")
        if header:
            return points[1:]
        else:
            return points

if __name__ == "__main__":
    mr = MyLogisticRegression("data/iris2.data", header=True)

    coeffs, rss = mr.run(learning_rate=0.00000002, num_iterations=1000)

    # preds = mr.predict(pd.read_csv("data/treino_clean.csv"))
    #
    # print(preds)



    print(coeffs)
    print(rss)