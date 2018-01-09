import random
import numpy as np
import math

class Classifier:
  def __init__(self, training_data):
    """
    The classifier is initilized by giving it training data. At initialization the classifier calls the train method, which starts training the classifier with the given data.
    * training_data contains a pair (L, class) where L[i] is a data point and class[i] denotes to which class the point belongs
    """
    
    # store the dimension of the data
    self.dim = len(training_data[0][0])
    
    # store the number of classes
    self.cl = max(training_data[1])+1
    
    # store the number of data points 
    self.sample_size = len(training_data[0])
    
    self.training_data = training_data
    
    # initialize the weights and biases with random values
    self.weights = [[random.gauss(0, 1) for k in range(0, self.dim)] for j in range(0, self.cl)]
    self.biases = [random.gauss(0, 1) for j in range(0, self.cl)]
    
    self.train()

  def train(self):
    self.stoc_grad_mini_batch(10, 0.001)
    
  def loss_function(self):
    """
    Computes a cross-entropy loss function after softmax transform for testing purposes. Slow to evaluate when the data set is large.
    """
    L = 0.0
    for i in range(0, len(self.training_data[0])):
      r = 0.0
      for q in range(0, self.cl):
        r += math.exp(np.dot(self.weights[q], np.transpose(self.training_data[0][i])) + self.biases[q])
      L += math.log(r) - (np.dot(self.weights[self.training_data[1][i]], np.transpose(self.training_data[0][i])) + self.biases[self.training_data[1][i]])
    return L    

  def stoc_grad_mini_batch(self, batch_size = 10, errTarget = 0.001):
    """
    Outputs the gradient when used only some of the training data points, i.e., mini batch gradient descent. Uses the Adaptive Moment Estimation (Adam) heuristic for the learning rate.
    * batch_size gives how many data points we use for computing the mini batch gradient
    * errTarget gives the accepted error in the L1 distance of two consecutive solutions
    """
    
    # supresses output if set to False
    show_progress = True
    # if show_progress is set to True, output is shown every show_progress_iterations
    show_progress_iterations = 500
    
    weights_grad = [[0.0 for k in range(0, self.dim)] for j in range(0, self.cl)]
    biases_grad = [0.0 for j in range(0, self.cl)]
    
    def get_grad_batch(self):
      """ a subroutine that outputs a gradient w.r.t. a small random subset of the training data """

      # calculate a random set of indices in the range 1,...,len(training_data)
      idx = set([])
      while len(idx) < batch_size:
        idx.add(random.randint(0, len(self.training_data[0])-1))
    
      
      for i in idx:
        """ 
        pre-compute some coefficients that are needed in calculating the gradient 
        coeffs[j] = exp(v(j, i)), where v(q,i) = weights_q^T * x_i + bias_q        
        """
        coeffs = [math.exp(np.dot(self.weights[q], np.transpose(self.training_data[0][i])) + self.biases[q]) for q in range(0, self.cl)]
        
        # denom_normalization = sum w.r.t. q = 0,...,cl-1 of terms exp(v(q,i))
        denom_normalization = sum(coeffs)
      
        # normalize coeffs
        for c in range(0, len(coeffs)):
          coeffs[c] /= denom_normalization
      
        # calculate gradient w.r.t. weight[k][m] and then biases[k]
        for k in range(0, self.cl):
          for m in range(0, self.dim):
            weights_grad[k][m] += self.training_data[0][i][m]*coeffs[k]
            if self.training_data[1][i] == k:
              weights_grad[k][m] -= self.training_data[0][i][m]
       
          # calculate gradient w.r.t. bias k
          biases_grad[k] += coeffs[k]
          if self.training_data[1][i] == k:
            biases_grad[k] -= 1
        
        # normalize gradients (average over the batch size)
        for k in range(0, self.cl):
          for m in range(0, self.dim):
            weights_grad[k][m] /= batch_size
          biases_grad[k] /= batch_size
        
      return weights_grad, biases_grad
    
    # parameters and variables for the Adam heuristic
    beta1 = 0.9; beta2 = 0.999; eps = 10**(-8) 
    err = 10**4; step = 0.001
    
    # the mean variables represent the mean of the gradient (captures the notion of momentum)
    mean_weights = [[0.0 for k in range(0, self.dim)] for j in range(0, self.cl)]
    mean_biases = [0 for k in range(0, self.cl)]
    # the variance variables represent the variance of the gradient
    var_weights = [[0.0 for k in range(0, self.dim)] for j in range(0, self.cl)]
    var_biases = [0 for k in range(0, self.cl)]
    
    # variable to keep a counter on iterations in case we want to output data from the calculations
    time = 0
    # variable to stop the gradient descent after too many iterations
    max_iterations = 10**6
    # variables to speed up some computations
    b1 = 1; b2 = 1
    
    # main loop of the gradient descent iteration
    while err > errTarget and time < max_iterations:
      time += 1
      # grad_len keeps track of the L1 norm of the gradient, used only for testing
      grad_len = 0.0
      err = 0.0
      # compute a new gradient for weights and biases
      w_grad, b_grad = get_grad_batch(self);
      
      # record the current weights and biases for calculating how far we move during the iteration - implementation can be done without recording these, but we leave it here for clarity
      weights_prev = [[self.weights[i][j] for j in range(0, self.dim)] for i in range(0, self.cl)]
      biases_prev = list(self.biases)
            
      # having b1, b2 allows us to skip calculating beta1**time and beta2**time repeatedly (see the Adam procedure)
      b1 = b1*beta1; b2 = b2*beta2
      
      for k in range(0, self.cl):
        for m in range(0, self.dim):
          """
          compute the Adam means and variances for weights
          """
          mean_weights[k][m] = beta1*mean_weights[k][m] + (1-beta1)*w_grad[k][m]
          var_weights[k][m] = beta2*var_weights[k][m] + (1-beta2)*w_grad[k][m]*w_grad[k][m]
          est_m_weights = mean_weights[k][m]/(1-b1)
          est_v_weights = var_weights[k][m]/(1-b2)
          
          # move in the direction of the negative gradient
          self.weights[k][m] -= step*est_m_weights/(math.sqrt(est_v_weights)+eps)
          err += abs(self.weights[k][m]-weights_prev[k][m])
          grad_len += abs(w_grad[k][m])
        """
        compute the Adam means and variances for biases
        """
        mean_biases[k] = beta1*mean_biases[k] + (1-beta1)*b_grad[k]
        var_biases[k] = beta2*var_biases[k] + (1-beta2)*b_grad[k]*b_grad[k]
        est_m_biases = mean_biases[k]/(1-b1)
        est_v_biases = var_biases[k]/(1-b2)

        # move in the direction of the negative gradient
        self.biases[k] -= step*est_m_biases/(math.sqrt(est_v_biases)+eps)
        err += abs(self.biases[k]-biases_prev[k])
        grad_len += abs(b_grad[k])

      # optional step to print progress
      if show_progress == True and time % show_progress_iterations == 0:  
        print "Gradient descent error: ", err, ", loss function: ", self.loss_function(), ", gradient L1 norm: ", grad_len
      
  def classify(self, data_point):
    """Attempts to classify the given data point"""
    mProd = -10.0**10; mIdx = -1;
    # check which inner product with the weights + bias gives the highest score and return the corresponding index
    for i in range(0, self.cl):
      res = np.dot(self.weights[i], np.transpose(data_point)) + self.biases[i]; res = res[0]
      if res > mProd:
        mProd = res; mIdx = i
    return mIdx
  

class SampleData:
  """
  class SampleData generates normally distributed data points for testing the classifier
  """
  def __init__(self, dim, cl, mean_range = [-1, 1], variance_range = [0.5, 1]):
    """
    * dim - the dimension of the data (number of features)
    * cl - number of classes
    * mean_range - means of the generated data are drawn from a uniform distibution given by mean_range
    * variance_range - variances of the generated data are drawn from a uniform distibution given by mean_range
    """
    # self.mean - dim x cl dimensional matrix that holds the dim -dimensional mean vectors for each of the cl classes
    self.mean = [[random.uniform(mean_range[0], mean_range[1]) for k in range(0, dim)] for j in range(0, cl)]
    # self.variance - similarly as mean 
    self.variance = [[random.uniform(variance_range[0], variance_range[1]) for k in range(0, dim)] for j in range(0, cl)]
    self.dim = dim
    self.cl = cl
    
  def get_points(self, n):
    """
     Samples points from the distributionReturn value:
     * List of n points of dimension d. Each point is drawn from a multi-dimensional normal distribution where the mean and variance are uniformly sampled from pre-sampled list of means and variances
     * List of indices denoting from which class the point was drawn (for training the model)
    """
    L = []
    idx = []
    for i in range(0, n):
      idx.append(random.randint(0, self.cl-1))
      L.append([random.gauss(self.mean[idx[-1]][k], self.variance[idx[-1]][k]) for k in range(0, self.dim)])
      
    return L, idx
    
  
def main():
  """
  Simple tester for the classifier
  """
  # definte the sample data
  sd = SampleData(2, 5, [-3, 3], [0.1, 0.1])

  # initialize a new classifier with some sample data
  clf = Classifier(sd.get_points(500)) 
  
  # sample some points and see how many the classifier classifies correctly
  sample_size = 50; correct = 0
  for k in range(0, sample_size):
    p, i = sd.get_points(1); i = i[0]
    if clf.classify(p) == i:
      correct += 1
 
  print "Classified ", correct, " of ", sample_size, " correctly."

  
if __name__ == '__main__':
  main()