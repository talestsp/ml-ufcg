from numpy import *

def compute_error_for_given_function(b, m, array_points):
    total_error = 0
    for i in range(len(array_points)):
        x = array_points[i, 0]
        y = array_points[i, 1]
        total_error += (y - (m * x + b)) **2

    return total_error / float(len(array_points))

def step_gradient(b_current, m_current, array_points, learning_rate):
    #gradient_descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(array_points))

    for i in range(len(array_points)):
        x = array_points[i, 0]
        y = array_points[i, 1]

        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)

    return [new_b, new_m]


def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        rss = compute_error_for_given_function(b=b, m=m, array_points=points)
        print("Current RSS:", rss) #item 2
        print("b:", b)
        print("m:", m)
        print()

    print("---\nFinal RSS:", rss)

    return [b, m]


def run(datafile, learning_rate=0.0001, num_iterations=1000):
    points = genfromtxt(datafile, delimiter=",")

    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0

    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("b:", b)
    print("m:", m)
    print()



if __name__ == "__main__":
    # print("x = hours os study, y = test score")
    # run("data.csv", learning_rate=0.0001, num_iterations=1000)


    # #item 1
    print("x = years of formal education, y = money income")
    run("income.csv", learning_rate=0.001, num_iterations=47175) #item 4


    #item 3 - o RSS diminui pois o gradiente quase sempre (obs1) aponta para o ponto no espaço que possui RSS menor.
    #obs1: a menos que atinja o ponto ótimo