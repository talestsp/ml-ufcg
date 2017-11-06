from numpy import append, genfromtxt, array, ndarray
import pandas as pd

class MyRegression():

    def __init__(self, data, header=False):
        if type(data) == ndarray:
            if header:
                data = data[1:]
            else:
                data = data
            self.data = data
        elif type(data) == str:
            self.data = self.load_data(data, header)

    def compute_error_for_given_function(self, w_array, array_points):
        total_error = 0
        for i in range(len(array_points)):
            tuple_len = len(array_points[i])
            x_array = append(array(1), array_points[i, 0:tuple_len - 1])
            y = array_points[i, tuple_len - 1]

            total_error += (y - (self.hypothesis(w_array, x_array))) ** 2

        return total_error / float(len(array_points))

    def hypothesis(self, w_array, x_array):
        assert len(w_array) == len(x_array)

        total = 0
        for i in range(len(w_array)):
            total += w_array[i] * x_array[i]

        return total


    def step_gradient(self, current_w_array, array_points, learning_rate):
        # gradient_descent
        w_gradient_array = [0] * array_points.shape[1]
        N = float(len(array_points))

        for i in range(len(array_points)):
            tuple_len = len(array_points[i])

            x_array = append(array(1), array_points[i, 0:tuple_len - 1])
            y = array_points[i, tuple_len - 1]

            for i in range(len(w_gradient_array)):
                w_gradient_array[i] += self.gradiente_descent_calc(current_w_array, x_array, N, y, x_array[i])

        # update coefficients
        new_w_array = [None] * len(current_w_array)
        for i in range(len(current_w_array)):
            new_w_array[i] = current_w_array[i] - (2 * learning_rate * w_gradient_array[i])

        return new_w_array


    def gradiente_descent_calc(self, current_w_array, x_array, N, y, x):
        assert len(current_w_array) == len(x_array)

        return -1 * (y - (self.hypothesis(current_w_array, x_array))) * x


    def gradient_descent_runner(self, points, initial_w_array, learning_rate, num_iterations, cost_tolerance, verbosity=False):
        w_array = initial_w_array

        rss = self.compute_error_for_given_function(w_array, array_points=points)

        iterations_count = 0
        while rss >= cost_tolerance and iterations_count <= num_iterations:
            iterations_count += 1

            w_array = self.step_gradient(w_array, array(points), learning_rate)

            rss = self.compute_error_for_given_function(w_array, array_points=points)

            if verbosity == "vv":
                print("Current RSS:", rss)  # item 2

        if verbosity == "v" or verbosity == "vv":
            print()
            print("---\nFinal RSS:", rss)
        return w_array, rss

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

        self.w_array, self.rss = self.gradient_descent_runner(self.data, initial_w_array, learning_rate, num_iterations, cost_tolerance,
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
    mr = MyRegression("data/treino_clean.csv", header=True)


    coeffs, rss = mr.run(learning_rate=0.00000002, num_iterations=1000)

    preds = mr.predict(pd.read_csv("data/treino_clean.csv"))

    print(preds)



    print(coeffs)
    print(rss)