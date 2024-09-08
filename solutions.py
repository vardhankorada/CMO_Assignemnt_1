'''

*************CMO ASSIGNMENT 1 SUBMISSION*********

Run the file as "python solutions.py > output.txt"
The output will be written to output.txt
All the figires will be store in the parent directory only.

'''



'''
IMPORTS
'''
from CMO_A1 import f1,f2,f3,f4
import numpy as np
import matplotlib.pyplot as plt



'''
GLOBAL CONSTANTS
'''
srno = 24380
precision = 7



'''
FUNCTIONS ASKED IN THE ASSIGNMENT
'''
def isConvex(f,interval):
    a,b = interval[0],interval[1]
    print("Sampling function to understand the underlying structure of the function")
    delta = 0.01
    sample_points = np.array([np.round(a+(delta*i),3) for i in np.arange((b-a)/delta + 1)])
    sample_fvals = np.array([np.round(f(srno,x),precision) for x in sample_points])
    plt.clf()
    plt.plot(sample_points,sample_fvals,'.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("Samples_of_the_function_{}.png".format(f.__name__))
    print("Checking the base convex function definition")
    c,sc = True, True
    alphas = [i*0.01 for i in range(101)]
    num_points = 100
    points = [a + i * (b - a) / num_points for i in range(num_points + 1)]
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            x_1, x_2 = points[i], points[j]
            for alpha in alphas:
                lhs = np.round(f(srno,alpha * x_1 + (1-alpha) * x_2),precision) 
                rhs = np.round(alpha * f(srno,x_1) + (1-alpha) * f(srno,x_2),precision)
                if lhs > rhs:
                    c = False
                if alpha not in [0,1] and lhs >= rhs:
                    sc = False
    f_min = np.min(sample_fvals)
    indices = np.array([int(x) for x in np.where(sample_fvals == f_min)[0]])
    x_min = sample_points[indices]
    return (c,sc,f_min,x_min)

def isCoercive(f):
    print("Sampling function to understand the underlying structure of the function")
    sample_points_p = [10**i for i in range(0,5)]
    f_vals_p = [np.round(f(srno,x),precision) for x in sample_points_p]
    sample_points_n = [-10**i for i in range(0,5)]
    f_vals_n = [np.round(f(srno,x),precision) for x in sample_points_n]
    f_vals_n.extend(f_vals_p)
    sample_points_n.extend(sample_points_p)
    plt.clf()
    plt.plot(sample_points_n,f_vals_n,"x")
    plt.savefig("Samples_of_the_function_{}.png".format(f.__name__))
    print("Checking function value trend as ||x|| --> infinity")
    threshold = 10**3
    x_p = [threshold + 10**i for i in range(1,5)]
    x_n = [-threshold-10**i for i in range(1,5)]
    ans_p = [f(srno,x) for x in x_p]
    ans_n = [f(srno,x) for x in x_n]
    return sorted(ans_p) == ans_p and sorted(ans_n) == ans_n

def FindStationaryPoints(f):
    def find_initial_intervals(f, start, end, step):
        intervals = []
        x_prev = start
        f_prev = f(srno,x_prev)
        for x in np.arange(start, end, step):
            f_curr = f(srno,x)
            if f_prev * f_curr < 0:
                intervals.append((x_prev, x))
            x_prev, f_prev = x, f_curr
        return intervals
    def bisection_method(f,interval,tol=1e-3):
        a,b = interval
        while (b - a) / 2 > tol:
            mid = (a + b) / 2
            if f(srno,mid) == 0:
                return mid
            elif f(srno,a) * f(srno,mid) < 0:
                b = mid
            else:
                a = mid
        return (a + b) / 2
    def trisection_method(h, interval, tol=1e-3):
        a,b = interval
        while (b - a) > tol:
            x1 = a + (b - a) / 3
            x2 = b - (b - a) / 3
            f_x1 = h(srno,x1)
            f_x2 = h(srno,x2)
            if f_x1 < f_x2:
                b = x2
            else:
                a = x1
        return (a + b) / 2
    def g(srno,x): return -1*f(srno,x)
    print("Sampling function to understand the underlying structure of the function")
    a,b,delta,h = -3,3,0.01,1e-2   ## Tinkered these values until #roots = 4
    sample_points = np.array([np.round(a+(delta*i),3) for i in np.arange((b-a)/delta + 1)])
    sample_fvals = np.array([np.round(f(srno,x),precision) for x in sample_points])
    plt.clf()
    plt.plot(sample_points,sample_fvals,'.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("Samples_of_the_function_{}.png".format(f.__name__))
    print("Using bisection and trisection methods to find the roots and extremums")
    roots,mins,maxs = [],[],[]
    initial_intervals = find_initial_intervals(f, a, b, delta)
    for interval in initial_intervals:
        roots.append(np.round(bisection_method(f,interval),precision))
    roots = sorted(roots)
    for i in range(1,len(roots)):
        interval = (roots[i-1],roots[i])
        temp_1,temp_2 = trisection_method(f,interval), trisection_method(g,interval)
        if f(srno,temp_1) < f(srno,temp_1-h) and f(srno,temp_1) < f(srno,temp_1+h):
            mins.append(np.round(temp_1,precision))
        if g(srno,temp_2) < g(srno,temp_2-h) and g(srno,temp_2) < g(srno,temp_2+h):
            maxs.append(np.round(temp_2,precision))
    res = {"Roots":roots,"Minima":mins,"LocalMaxima":maxs}
    return res

def ConstantGradientDescent(alpha,initialx):
    tolerance = 1e-5
    f_k,g_k = f4(srno,initialx)
    x_k = initialx
    iter_count = 0
    track = [(x_k,f_k,g_k)]
    while np.linalg.norm(g_k) >= tolerance:
        x_k = x_k - alpha * g_k
        f_k,g_k = f4(srno,x_k)
        iter_count += 1
        track.append((x_k,f_k,g_k))
    return track

def DiminishingGradientDescent(InitialAlpha, initialx):
    min_iters = 10000
    f_k,g_k = f4(srno,initialx)
    x_k = initialx
    iter_count = 0
    tolerance = 1e-5
    track = [(x_k,f_k,g_k)]
    while iter_count < min_iters-1 and np.linalg.norm(g_k) >= tolerance:
        alpha = InitialAlpha/(iter_count+1)
        x_k = x_k - alpha * g_k
        f_k,g_k = f4(srno,x_k)
        iter_count += 1
        track.append((x_k,f_k,g_k))
    return track

def InExactLineSearch(c1,c2, gamma):
    def condition_1(alpha_k,x_k,c1):
        f_k,g_k = f4(srno,x_k)
        n_x = x_k - alpha_k * g_k
        f_n_x,_ = f4(srno,n_x)
        return f_n_x <= f_k - c1*alpha_k*np.dot(g_k,g_k)
    def condition_2(alpha_k,x_k,c2):
        _,g_k = f4(srno,x_k)
        n_x = x_k - alpha_k * g_k
        _,g_n_x = f4(srno,n_x)
        return np.dot(g_k,g_n_x) <= c2*np.dot(g_k,g_k)
    def choose_alpha(x_k,c1,c2,gamma):
        alpha_k = 1
        while not(condition_1(alpha_k,x_k,c1) and condition_2(alpha_k,x_k,c2)):
            alpha_k = gamma * alpha_k
        return alpha_k
    tolerance = 1e-4
    initialx = np.array([0.0,0.0,0.0,0.0,0.0])
    f_k,g_k = f4(srno,initialx)
    x_k = initialx
    iter_count = 0
    track = [(x_k,f_k,g_k)]
    while np.linalg.norm(g_k) >= tolerance:
        x_k = x_k - choose_alpha(x_k,c1,c2,gamma) * g_k
        f_k,g_k = f4(srno,x_k)
        iter_count += 1
        track.append((x_k,f_k,g_k))
    return track

def ExactLineSearch():
    def calc_alpha(x_k,b):
        _,g_k = f4(srno,x_k)
        num = np.dot(g_k,g_k)
        f_g_k,_ = f4(srno,g_k)
        den = 2*(f_g_k - np.dot(b,g_k))
        return num/den
    initialx = np.array([0.0,0.0,0.0,0.0,0.0])
    tolerance = 1e-5
    f_k,g_k = f4(srno,initialx)
    b = g_k
    x_k = initialx
    iter_count = 0
    track = [(x_k,f_k,g_k)]
    while np.linalg.norm(g_k) >= tolerance:
        x_k = x_k - calc_alpha(x_k,b) * g_k
        f_k,g_k = f4(srno,x_k)
        iter_count += 1
        track.append((x_k,f_k,g_k))
    return track



'''
HELPER FUNCTIONS
'''
def plot_track(res,name):
    norms = np.array([np.linalg.norm(x[2]) for x in res])
    f_T,x_T = res[-1][1],res[-1][0]
    f_diff = np.array([res[i][1]-f_T for i in range(len(res))])
    f_ratio = np.array([f_diff[i]/f_diff[i-1] for i in range(1,len(f_diff-1))])
    x_diff = np.array([np.linalg.norm(res[i][0]-x_T)**2 for i in range(len(res))])
    x_ratio = np.array([x_diff[i]/x_diff[i-1] for i in range(1,len(x_diff-1))])
    plt.clf()
    plt.plot(np.arange(0,len(norms)),norms)
    plt.title("Grad Norm")
    plt.savefig("Grad_Norm_{}.png".format(name))
    plt.clf()
    plt.plot(np.arange(0,len(f_diff)),f_diff)
    plt.title("Function Val Distance")
    plt.savefig("Func_val_dist_{}.png".format(name))
    plt.clf()
    plt.plot(np.arange(0,len(f_ratio)),f_ratio)
    plt.title("Function Val Dist. Ratio")
    plt.savefig("Func_val_dist_ratio_{}.png".format(name))
    plt.clf()
    plt.plot(np.arange(0,len(x_diff)),x_diff)
    plt.title("X Val Distance")
    plt.savefig("x_val_dist_{}.png".format(name))
    plt.clf()
    plt.plot(np.arange(0,len(x_ratio)),x_ratio)
    plt.title("X Val Dist. Ratio")
    plt.savefig("x_val_dist_ratio_{}.png".format(name))

def p(x,y) : return np.exp(x*y)

def get_grad(x,y):
    df_dx = y * np.exp(x*y)
    df_dy = x * np.exp(x*y)
    return np.array([df_dx,df_dy])

def plot_track_two(res,fvals,num_iter,name):
    print("Number of iters = {} || Final function value = {} || Final coordinates = {}".format(num_iter,fvals[-1],res[-1]))
    print("---> Plotting")
    x = np.linspace(-1,1,100)
    y = np.linspace(-1,1,100)
    X, Y = np.meshgrid(x, y)
    vals = p(X,Y)
    plt.clf()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.contour(X, Y, vals, levels=20, cmap='viridis')
    plt.plot(res[:, 0], res[:, 1], marker='o', color='red', label='Trajectory')
    plt.scatter(res[0, 0], res[0, 1], color='green', s=100, label='Start')
    plt.scatter(res[-1, 0], res[-1, 1], color='blue', s=100, label='End')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent Trajectory on Contour Plot')
    plt.colorbar(label='f(x, y) = e^(xy)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(num_iter + 1), fvals, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Function Value at Each Iteration')
    plt.grid()
    plt.tight_layout()
    plt.savefig("Contour_plot_{}.png".format(name))

def q(x): return x*(x-1)*(x-3)*(x+2)

def GSS(f,interval,tol=1e-4):
    a,b = interval
    phi = (1 + np.sqrt(5)) / 2
    rho = phi - 1
    x1 = rho * a + (1 - rho) * b
    x2 = (1 - rho) * a + rho * b
    res = [[a,b]]
    f_x1, f_x2 = f(x1), f(x2)
    while (b-a) >= tol:
        if f_x1 <= f_x2:
            b = x2
            x2 = x1
            x1 = rho * a + (1-rho) * b
        else:
            a = x1
            x1 = x2  
            x2 = (1-rho) * a + rho * b  ## CHANGED FROM THE GIVEN
        f_x1, f_x2 = f(x1), f(x2)
        res.append([a,b])
    return res

def FSS(f,interval,tol=1e-4):
    def gen_fib_nums(x):
        res = [0,1]
        for i in range(x):
            res.append(res[-1]+res[-2])
        return res
    a,b = interval
    res = [[a,b]]
    req_num_iters = 21
    fib_nums = gen_fib_nums(req_num_iters+5)[1:]
    curr_iter = 1
    while (b-a) >= tol:
        rho = 1 - (fib_nums[req_num_iters-curr_iter]/fib_nums[req_num_iters-curr_iter+2]) ## MODIFIED
        if curr_iter == 1:
            x1 = rho * a + (1 - rho) * b
            x2 = (1 - rho) * a + rho * b
            f_x1, f_x2 = f(x1), f(x2)
        if f_x1 <= f_x2:
            b = x2
            x2 = x1
            x1 = rho * a + (1-rho) * b
        else:
            a = x1
            x1 = x2
            x2 = (1-rho) * a + rho * b ## MODIFIED
        f_x1, f_x2 = f(x1), f(x2)
        res.append([a,b])
        curr_iter += 1
    return res



'''
QUESTION 1
'''
def q1_a():
    def check_convexity(f,name):
        print("*******Checking convex nature of {}*******".format(name))
        (c,sc,fs,xs) = isConvex(f,(-2,2))
        print("--> The convexity of the funciton is {} anf strict convexity of the function is {}".format(c,sc))
        if len(xs) > 1:
            print("--> There are multiple minima for the function and the function value at those minima is {}".format(fs))
            print("--> The minima are approximately between {} and {}".format(xs[0],xs[-1]))
        else:
            print("--> There is an unique minima for the function and the function value at that minima is {}".format(fs))
            print("--> The minima is {}".format(xs[0]))
    check_convexity(f1,"f1")
    check_convexity(f2,"f2")

def q1_b():
    print("*******Checking the coercitivity of f3*******")
    print("--> The coercitivity of the function is {}".format(isCoercive(f3)))
    print("*******Finding the stationary points of f3*******")
    res = FindStationaryPoints(f3)
    print("--> Roots : {}".format(res["Roots"]))
    print("--> Minima : {}".format(res["Minima"]))
    print("--> LocalMaxima : {}".format(res["LocalMaxima"]))



'''
QUESTION 2
'''
def q2_a():
    print("*******Performing constant gradient*******")
    res = ConstantGradientDescent(1e-5,np.array([0.0,0.0,0.0,0.0,0.0]))
    print("---> Number of iterations = {}".format(len(res)))
    print("---> Final \n x_k = {} \n f_k = {} \n g_k = {} \n norm(g_k) = {}".format(res[-1][0],res[-1][1],res[-1][2],np.linalg.norm(res[-1][2])))
    print("10000th \n x_k = {} \n f_k = {} \n g_k = {} \n norm(g_k) = {}".format(res[9999][0],res[9999][1],res[9999][2],np.linalg.norm(res[9999][2])))
    print("---> Plotting curves <---")
    plot_track(res,"q2_a")

def q2_b():
    print("*******Performing dimnishing gradient*******")
    res = DiminishingGradientDescent(1e-3,np.array([0.0,0.0,0.0,0.0,0.0]))
    print("---> Number of iterations = {}".format(len(res)))
    print("---> Final \n x_k = {} \n f_k = {} \n g_k = {} \n norm(g_k) = {}".format(res[-1][0],res[-1][1],res[-1][2],np.linalg.norm(res[-1][2])))
    print("---> Plotting curves <---")
    plot_track(res,"q2_b")

def q2_c():
    print("*******Performing inexact line_search*******")
    res = InExactLineSearch(c1=1e-4,c2=1-(1e-4),gamma=0.9)
    print("---> Number of iterations = {}".format(len(res)))
    print("---> Final \n x_k = {} \n f_k = {} \n g_k = {} \n norm(g_k) = {}".format(res[-1][0],res[-1][1],res[-1][2],np.linalg.norm(res[-1][2])))
    print("---> Plotting curves <---")
    plot_track(res,"q2_c")

def q2_d():
    print("*******Performing exact line_search*******")
    res = ExactLineSearch()
    print("---> Number of iterations = {}".format(len(res)))
    print("---> Final \n x_k = {} \n f_k = {} \n g_k = {} \n norm(g_k) = {}".format(res[-1][0],res[-1][1],res[-1][2],np.linalg.norm(res[-1][2])))
    print("---> Plotting curves <---")
    plot_track(res,"q2_d")



'''
QUESTION 3
'''
def q3_3():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    vals = p(X,Y)
    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, vals, levels=20, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8) 
    plt.colorbar(label='f(x, y) = e^(xy)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Contour Plot of f(x, y) = e^(xy)')
    plt.grid()
    plt.savefig("Contour_plot_q3_3.png")

def q3_4():
    def perf_grad_desc(step_size = 0.1,start=(1,1)):
        curr_x,curr_y = start
        tolerance = 1e-5
        grad = get_grad(curr_x,curr_y)
        f_val = p(curr_x,curr_y)
        res = [[curr_x,curr_x]]
        fvals = [f_val]
        num_iter = 0
        while np.linalg.norm(grad) >= tolerance:
            curr_x -= step_size * grad[0]
            curr_y -= step_size * grad[1]
            grad = get_grad(curr_x,curr_y)
            f_val = p(curr_x,curr_y)
            res.append([curr_x,curr_y])
            fvals.append(f_val)
            num_iter += 1
        res = np.array(res)
        fvals = np.array(fvals)
        return res,fvals,num_iter
    print("*******Performing constant step grad descent*******")
    res,fvals,num_iter = perf_grad_desc(0.1,(1,1))
    plot_track_two(res,fvals,num_iter,"q3_4")

def q3_5():
    def perf_grad_desc(step_size = 0.1,start=(1,1)):
        curr_x,curr_y = start
        tolerance = 1e-2
        grad = get_grad(curr_x,curr_y)
        f_val = p(curr_x,curr_y)
        res = [[curr_x,curr_x]]
        fvals = [f_val]
        num_iter = 0
        while np.linalg.norm(grad) >= tolerance:
            curr_x -= step_size * grad[0]
            curr_y -= step_size * grad[1]
            grad = get_grad(curr_x,curr_y)
            f_val = p(curr_x,curr_y)
            res.append([curr_x,curr_y])
            fvals.append(f_val)
            num_iter += 1
            step_size *= 0.995
        res = np.array(res)
        fvals = np.array(fvals)
        return res,fvals,num_iter
    print("*******Performing dimnishing step grad descent*******")
    res,fvals,num_iter = perf_grad_desc(0.1,(1,1))
    plot_track_two(res,fvals,num_iter,"q3_5")

def q3_6():
    def perf_grad_desc(step_size = 0.1,start=(1,1)):
        curr_x,curr_y = start
        tolerance = 1e-5
        grad = get_grad(curr_x,curr_y)
        f_val = p(curr_x,curr_y)
        res = [[curr_x,curr_x]]
        fvals = [f_val]
        num_iter = 0
        sigma = 0.1
        while np.linalg.norm(grad) >= tolerance:
            noise_samples = np.random.normal(0, sigma, (5, 2))
            mean_noise = np.mean(noise_samples, axis=0)
            curr_x -= step_size * (grad[0] + mean_noise[0]) 
            curr_y -= step_size * (grad[1] + mean_noise[1])
            grad = get_grad(curr_x,curr_y)
            f_val = p(curr_x,curr_y)
            res.append([curr_x,curr_y])
            fvals.append(f_val)
            num_iter += 1
        res = np.array(res)
        fvals = np.array(fvals)
        return res,fvals,num_iter
    print("*******Performing Pertubated constant step and constant variance grad descent*******")
    res,fvals,num_iter = perf_grad_desc(0.1,(1,1))
    plot_track_two(res,fvals,num_iter,"q3_6")

def q3_7():
    def perf_grad_desc(step_size = 0.1,start=(1,1)):
        curr_x,curr_y = start
        tolerance = 1e-5
        grad = get_grad(curr_x,curr_y)
        f_val = p(curr_x,curr_y)
        res = [[curr_x,curr_x]]
        fvals = [f_val]
        num_iter = 0
        sigma_init = 0.1
        sigma = sigma_init
        while np.linalg.norm(grad) >= tolerance:
            noise_samples = np.random.normal(0, sigma, (5, 2))
            mean_noise = np.mean(noise_samples, axis=0)
            curr_x -= step_size * (grad[0] + mean_noise[0]) 
            curr_y -= step_size * (grad[1] + mean_noise[1])
            grad = get_grad(curr_x,curr_y)
            f_val = p(curr_x,curr_y)
            res.append([curr_x,curr_y])
            fvals.append(f_val)
            num_iter += 1
            sigma = sigma_init/(num_iter+1)
        res = np.array(res)
        fvals = np.array(fvals)
        return res,fvals,num_iter
    print("*******Performing Pertubated constant step and decreasing variance grad descent*******")
    res,fvals,num_iter = perf_grad_desc(0.1,(1,1))
    plot_track_two(res,fvals,num_iter,"q3_7")

def q3_8():
    def perf_grad_desc(step_size = 0.1,start=(1,1)):
        curr_x,curr_y = start
        tolerance = 1e-3
        grad = get_grad(curr_x,curr_y)
        f_val = p(curr_x,curr_y)
        res = [[curr_x,curr_x]]
        fvals = [f_val]
        num_iter = 0
        step_size_init = 0.1
        sigma = 0.1
        while np.linalg.norm(grad) >= tolerance and num_iter < 5*10**4:
            noise_samples = np.random.normal(0, sigma, (5, 2))
            mean_noise = np.mean(noise_samples, axis=0)
            curr_x -= step_size * (grad[0] + mean_noise[0]) 
            curr_y -= step_size * (grad[1] + mean_noise[1])
            grad = get_grad(curr_x,curr_y)
            f_val = p(curr_x,curr_y)
            res.append([curr_x,curr_y])
            fvals.append(f_val)
            num_iter += 1
            step_size = step_size_init/(num_iter+1)
        res = np.array(res)
        fvals = np.array(fvals)
        return res,fvals,num_iter
    print("*******Performing Pertubated decreasing step and constant variance grad descent*******")
    res,fvals,num_iter = perf_grad_desc(0.1,(1,1))
    plot_track_two(res,fvals,num_iter,"q3_8")

def q3_9():
    def perf_grad_desc(step_size = 0.1,start=(1,1)):
        curr_x,curr_y = start
        tolerance = 1e-3
        grad = get_grad(curr_x,curr_y)
        f_val = p(curr_x,curr_y)
        res = [[curr_x,curr_x]]
        fvals = [f_val]
        num_iter = 0
        step_size_init = 0.1
        step_size = step_size_init
        sigma_init = 0.1
        sigma = sigma_init
        while np.linalg.norm(grad) >= tolerance and num_iter < 5 * 10**4:
            noise_samples = np.random.normal(0, sigma, (5, 2))
            mean_noise = np.mean(noise_samples, axis=0)
            curr_x -= step_size * (grad[0] + mean_noise[0]) 
            curr_y -= step_size * (grad[1] + mean_noise[1])
            grad = get_grad(curr_x,curr_y)
            f_val = p(curr_x,curr_y)
            res.append([curr_x,curr_y])
            fvals.append(f_val)
            num_iter += 1
            step_size = step_size_init/(num_iter+1)
            sigma = sigma_init/(num_iter+1)
        res = np.array(res)
        fvals = np.array(fvals)
        return res,fvals,num_iter
    print("*******Performing Pertubated decreasing step and constant variance grad descent*******")
    res,fvals,num_iter = perf_grad_desc(0.1,(1,1))
    plot_track_two(res,fvals,num_iter,"q3_9")



'''
QUESTION 4
'''
def q4_a():
    print("*******Performing Golden Section Search*******")
    res = GSS(q,(1,3),1e-4)
    print("Intervals : ",res)
    print("---> Final interval is {}".format(res[-1]))
    f_a,f_b = [np.round(q(tup[0]),5) for tup in res],[np.round(q(tup[1]),5) for tup in res]
    diff = [np.round(tup[1]-tup[0],5) for tup in res]
    ratio = [np.round(diff[i]/diff[i-1],5) for i in range(1,len(diff))]
    plt.clf()
    plt.plot(np.arange(0,len(res)),f_a)
    plt.title("f(a_t) vs t")
    plt.savefig("f_a_t_vs_t_q4_a.png")
    plt.clf()
    plt.plot(np.arange(0,len(res)),f_b)
    plt.title("f(b_t) vs t")
    plt.savefig("f_b_t_vs_t_q4_a.png")
    plt.clf()
    plt.plot(np.arange(0,len(res)),diff)
    plt.title("(b_t - a_t) vs t")
    plt.savefig("b_t_-_a_t_vs_t_q4_a.png")
    plt.clf()
    plt.plot(np.arange(0,len(ratio)),ratio)
    plt.title("(b_t-a_t)/(b_t-1 - a_t-1) vs t")
    plt.savefig("ratio_q4_a.png")

def q4_b():
    print("*******Performing Fibonacci Search*******")
    res = FSS(q,(1,3),1e-4)
    print("Intervals : ",res)
    print("---> Final interval is {}".format(res[-1]))
    f_a,f_b = [np.round(q(tup[0]),5) for tup in res],[np.round(q(tup[1]),5) for tup in res]
    diff = [np.round(tup[1]-tup[0],5) for tup in res]
    ratio = [np.round(diff[i]/diff[i-1],5) for i in range(1,len(diff))]
    plt.clf()
    plt.plot(np.arange(0,len(res)),f_a)
    plt.title("f(a_t) vs t")
    plt.savefig("f_a_t_vs_t_q4_b.png")
    plt.clf()
    plt.plot(np.arange(0,len(res)),f_b)
    plt.title("f(b_t) vs t")
    plt.savefig("f_b_t_vs_t_q4_b.png")
    plt.clf()
    plt.plot(np.arange(0,len(res)),diff)
    plt.title("(b_t - a_t) vs t")
    plt.savefig("b_t_-_a_t_vs_t_q4_b.png")
    plt.clf()
    plt.plot(np.arange(0,len(ratio)),ratio)
    plt.title("(b_t-a_t)/(b_t-1 - a_t-1) vs t")
    plt.savefig("ratio_q4_b.png")



'''
DRIVER FUNCTION
'''
def run_script():
    print("\n#######QUESTION 1#######\n")
    q1_a()
    q1_b()
    print("\n#######QUESTION 2#######\n")
    q2_a()
    q2_b()
    q2_c()
    q2_d()
    print("\n#######QUESTION 3#######\n")
    q3_3()
    q3_4()
    q3_5()
    q3_6()
    q3_7()
    q3_8()
    q3_9()
    print("\n#######QUESTION 4#######\n")
    q4_a()
    q4_b()


'''
MAIN FUNCTION 
'''
if __name__ == "__main__":
    run_script()