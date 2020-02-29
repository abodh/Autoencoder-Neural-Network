import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

#initialization of values
i = 8  # total number of input neurons
h = 4  # total number of hidden neurons
o = 8  # total number of output neurons
b = 1  # total number of biased neurons
eta = 0.3  # learning rate
wih = num.random.randn(i, h-b)/100  # random weights from input to hidden network
who = num.random.randn(h, o)/100    # random weights from hidden to output network
xi = num.identity(8)  # identity matrix of order 8
to = xi  # target output
#error_stack = num.ones((1, o))

#initialization for plotting
SSE_graph = num.ones((1, o)) #to get the initial value of SSE
hidden_graph = num.ones((1, h-b)) #to get the initial value of hidden neurons
wih_graph = num.ones((1, i)) #to get the initial value of input to hidden weight
final_output = num.ones((1,i)) #to get the initial vale of output to be displayed
final_hidden = num.ones((1,h-b)) #to get the final value of hidden output to be displayed

#learning process
for loop in range(1, 5000): #for total iteration
    sse_value = 0
    for x in range(0, 8): #for total process based on number of input values

        #forward pass of network

        net = num.matmul(xi[x, :], wih)  # 1x8 * 8x3 = 1x3 : summation(i) = wi * xi
        hidden_output = num.divide(1, (1 + num.exp(-net)))  # output value of hidden neurons
        hidden_output_b = num.hstack((hidden_output, 1)) # stacking the bias to the end
        net_output = num.matmul(hidden_output_b, who)  # 1x4 * 4x8 = 1x8 : summation(h) = wh * xh
        output = num.divide(1, (1 + (num.exp(-net_output))))  # 1x8 #output value of output neurons
        error = abs(num.array(to[x, :]) - num.array(output))

        # Back propagation

        # from output to hidden layer

        del_k = (output * (1 - output)) * (to[x, :] - output)  # 1x8
        del_who = eta * num.matmul(num.array([hidden_output_b]).T, num.array([del_k]))  # 3x1 * 1x8 = 3x8

        # from hidden to input layer

        del_h1 = num.matmul(del_k, (who[0:(len(who)-1), :]).transpose())  # 1x8 * (3x8)T = 1x3
        del_h2 = hidden_output * (1 - hidden_output)  # 1x4
        del_h = del_h1 * del_h2  # 1x4
        del_wih = eta * num.matmul(num.array([xi[x, :]]).T, num.array([del_h]))

        # Updating the weights

        wih = wih + del_wih;
        who = who + del_who;

        sqer = (to[x,:] - output) * (to[x,:] - output)
        sse_value = sse_value + sqer
        if x==1:
            hidden_graph = num.vstack((hidden_graph, hidden_output))
    SSE_graph = num.vstack((SSE_graph, sse_value))
    wih_graph = num.vstack((wih_graph, num.array([wih[:,0]])))

    #error_stack = num.vstack((error_stack, error))

#removing the original initialization for plotting

SSE_graph = SSE_graph[1:(len(SSE_graph)),:]
hidden_graph = hidden_graph [1:(len(hidden_graph)),:]
wih_graph = wih_graph [1:(len(wih_graph)),:]

# evaluation
for x in range(0, 8):
    net = num.matmul(xi[x, :], wih)  # 1x8 * 8x3 = 1x3 : summation(i) = wi * xi
    hidden_output = num.divide(1, (1 + num.exp(-net)))  # output value of hidden neurons
    hidden_output_b = num.hstack ((hidden_output, 1))
    net_output = num.matmul(hidden_output_b, who)  # 1x3 * 3x8 = 1x8 : summation(h) = wh * xh
    output = num.divide(1, (1 + (num.exp(-net_output))))  # 1x8 #output value of output neurons
    final_hidden = num.vstack((final_hidden, hidden_output))
    final_output = num.vstack((final_output, output))

final_hidden = final_hidden[1:(len(final_hidden)), :]
final_output = final_output[1:(len(final_output)), :]
print ("input value")
print (xi[:,:])
print ("hidden value")
print (final_hidden)
print ("output value")
print(final_output)

# Writing data to csv
final_output=pd.DataFrame(final_output,columns=['T1','T2','T3','T4','T5','T6','T7','T8'])
final_output.to_csv('Test_Data_out.csv', header=True)
final_hidden=pd.DataFrame(final_hidden,columns=['T1','T2','T3'])
final_hidden.to_csv('Test_Data_hid.csv', header=True)

item_no_sse = int(num.shape(SSE_graph)[1])
for i in range(0, item_no_sse):
    plt.plot(SSE_graph[:, i])

plt.grid()
plt.ylabel("Squared Error")
plt.xlabel("Number of iteration")
plt.title("Sum of squared error for each output units")
plt.xlim([0,5000])
plt.ylim([0,2])
plt.savefig ('SSE', dpi = 600)
plt.show()

item_no_weight = int(num.shape(wih_graph)[1])
for i in range(0, item_no_weight):
    plt.plot(wih_graph[:, i])

plt.grid()
plt.ylabel("weight value")
plt.xlabel("Number of iteration")
plt.title("Weights from inputs to one hidden unit")
plt.xlim([0,5000])
plt.ylim([-5,5])
plt.savefig ('Weights_i_h', dpi = 600)
plt.show()

item_no_hidden = int(num.shape(hidden_graph)[1])
for i in range(0, item_no_hidden):
     plt.plot(hidden_graph[:, i])

plt.grid()
plt.ylabel("output of hidden neuron")
plt.xlabel("Number of iteration")
plt.title("Hidden unit encoding for input 01000000")
plt.xlim([0,5000])
plt.ylim([0,2])
plt.savefig ('hidden_out', dpi = 600)
plt.show()