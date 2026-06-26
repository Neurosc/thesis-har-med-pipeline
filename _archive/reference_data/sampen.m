function se = sampen(x, m, r, tau)
%% Sample Entropy
% Entropy is a property of the state space of the system. Imagine a room
% full of gases. Let's assume there are N number of gas particles there. The
% state of the room can be characterized by a vector of 6N elements. Each
% gas particle has a position in 3 dimensional space and a velocity in
% those directions. Then, the entropy of this state at any time point can
% be calculated as S = -k * sum_i(p(i) log(p(i))) where k is the Boltzmann
% constant and sum_i means sum over i. p(i) is the probability of seeing
% i-th state. How to get those p's? Imagine plotting this 6 dimensional
% space (for simplification, let's imagine 2D where x axis is position
% and y axis is velocity in only one direction, higher dimensions work
% mathematically the same. Replace circles with spheres or whatever its 
% equivalent in 6 dimensions.). By drawing circles of radius epsilon, one 
% can build a probability distribution of seeing a particle in that circle i. 
% This is called coarse-graining. Let's say our first circle is at position 
% 3 and velocity 2. The probability of the first circle is the proportion 
% of particles having a position of 3 +- epsilon and a velocity of 2 +- 
% epsilon. Thus, we can also think in terms of histograms. The entropy can 
% be thought as a calculation made on the histogram itself. If we normalize 
% the histogram so that an integral over it = 1, then the formula can be 
% applied on height of each bar. 
% 
% The researchers working on electrophysiological (EEG / MEG) or
% haemodynamic (fMRI) data would want to apply this idea to time series. A
% brute application would be problematic. The problem is that the time
% points depend on each other. If one builds a histogram based on
% occurences of a coarse-grained version of a time series, one might
% realize that sampling from that distribution can not be random. The
% samples would have to depend on previous samples (since the time series
% correlates with itself, in other words, shows autocorrelation). To cross
% this problem, Richman and Moorman (2000, 
% https://doi.org/10.1152/ajpheart.2000.278.6.h2039) introduced sample
% entropy, an improvement on approximate entropy. This method uses a
% sophisticated algorithm to do coarse-graining and build the probability
% distribution. After the probability distribution is built, remaining
% entropy calculation is similar. 
% 
% Let's denote our time series as a. The procedure is as follows:
%   1) Set the tolerance parameter r. This serves as epsilon in our 
%      discussion above: How big our circles are.
%   2) Set the embedding dimension m. This is the part optimized for time
%      series data. The embedding dimension puts our time points in an
%      m-dimensional space. Again, let's consider 2 dimensions. x axis
%      would represent a(t) whereas y axis would represent a(t+tau). We can
%      put all the values in our time series to this two dimensional state
%      space. The dots will have the coordinates [a(t), a(t+tau)] for all
%      t. (If it was 3 dimensional, they would be [a(t), a(t+tau),
%      a(t+2tau)]). Do the same for a(t): that is, put all the a(t)s on a
%      line.
%   3) Then draw circles around each of these dots with radius r. The 
%      question is what is the proportion of dots being inside the first
%      plotting ([a(t), a(t+tau)]) over second (a(t)).
%   4) Then the sample entropy is -log2(A / B) where A is the number of
%      circles with 2 vectors inside in the first formulation and B is the 
%      number of circles with two dots inside the circles drawn on the line. Here, I
%      used log with base 2 to be in accordance with information
%      theoretical conventions.
% 
%  The above explanation works only for 2 dimensions (m = 2) and an
%  Euclidean distance (circle). Let's be more formal and generalize:
%  Build vectors X_m(i) = [x(i),  x(i+tau),  x(i+2tau), ..., x(i+(m-1)tau)].
%  Let d(X, Y) denote the distance function that calculates the distance
%  between vectors X and Y. In above circle example, we used Euclidean
%  distance. Below in the code, we will use Chebyshev distance which is
%  the minimum of elementwise differences between two functions. Finally,
%  the sample entropy is -log(A / B) where A = d(X_(m+1)(i), X_(m+1)(j)) <
%  r and B = d(X_m(i), X_m(j)) < r; i and j go over all t. So in essence,
%  how many dots are close enough in this high-dimensional space. 
% 
% Note the subtle introduction of time here. If there are time 
% dependencies, the dots will be close to each other with appropriate 
% selection of tau (low entropy). On the other hand, random / noisy time series' dots
% will be all over the places and the number of dots that are close will be
% low (high entropy). 
%
% Input:
%   x: time series (1D vector)
%   m: embedding dimension (integer, > 1)
%   r: tolerance parameter (scalar)
%   tau: time delays (integer, > 0)
%  Output:
%   se: sample entropy
% Example usage:
% se = sampen(x, 2, 0.2*std(x), 1)
% The code below is crude but easily readable for demonstration purposes. 
% For a more efficient and elegant implentation, see:
% https://github.com/MattWillFlood/EntropyHub/tree/main
% 
% Authored by Yasir Çatal
% catalyasir@gmail.com
%% Build different downsamplings of data based on tau
x = x(:);

if tau ~= 1
    xs = cell(1, tau);
    for i = 1:tau
        xs{i} = downsample(x(i:end), tau); % To cover all possible ways to build x(i+tau)
    end
else
    xs = {x};
end
%% Build a matrix of X_m

X_m = [];
X_m_plus_1 = [];
for t = 1:tau
    x_t = xs{t};
    N = length(x_t);
    % Build X_m
    for i = 1:(N-m+2)
        X_m(i, :, t) = x_t(i:(i+(m-2)));
    end
    % Now, X_(m+1)
    for i = 1:(N-m+1)
        X_m_plus_1(i, :, t) = x_t(i:(i+(m-1)));
    end
end

% Now we can start counting the number of occurrences for matrices X_m and
% X_(m+1)
% Initialize matrices (keep track of dimensions. dim 1 and dim 2 correspond 
% to i and j described above)
count_X_m = false(size(X_m, 1), size(X_m, 1), size(X_m, 3));
count_X_m_plus_1 = false(size(X_m_plus_1, 1), size(X_m_plus_1, 1), size(X_m_plus_1, 3));

for t = 1:tau
    % Start with X_m. d1 and d2 are two vectors to calculate the distance
    % between them. 
    for i = 1:size(X_m, 1)
        d1 = X_m(i, :, t);
        for j = (i+1):size(X_m, 1) % i+1: do not count twice
            d2 = X_m(j, :, t);
            count_X_m(i, j, t) = max(abs(d1 - d2)) < r; % Chebyshev distance < r
        end
    end
    % Do the same for X_(m+1)
    for i = 1:size(X_m_plus_1, 1)
        d1 = X_m_plus_1(i, :, t);
        for j = (i+1):size(X_m_plus_1, 1)
            d2 = X_m_plus_1(j, :, t);
            count_X_m_plus_1(i, j, t) = max(abs(d1 - d2)) < r; % Chebyshev distance < r
        end
    end
end

% Sum the number of counts
A = sum(count_X_m_plus_1(:));
B = sum(count_X_m(:));

se = -log2(A / B);
end





