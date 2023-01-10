#################################
# Your name: Tom Jakob
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals_list.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        sample = np.linspace(0, 1, m)
        contains = lambda x: False if (0.2 < x < 0.4) or (0.6 < x < 0.8) else True
        labels = np.zeros(m)
        for i in range(m):
            if contains(sample[i]):
                labels[i] = np.random.choice([0.0, 1.0], 1, p=[0.2, 0.8])
            else:
                labels[i] = np.random.choice([0.0, 1.0], 1, p=[0.9, 0.1])

        return np.column_stack((sample, labels))

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals_list.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        m_arr = np.arange(m_first, m_last + 1, step)
        empirical_err = np.zeros(len(m_arr))
        true_err = np.zeros(len(m_arr))
        for index, size in enumerate(m_arr):
            for i in range(T):
                sample = self.sample_from_D(size)
                x_axis, y_axis = sample[:, 0], sample[:, 1]
                interval_list, emp_err = intervals.find_best_interval(x_axis, y_axis, k)
                empirical_err[index] += emp_err
                true_err[index] += self.calc_true_err(interval_list)
            empirical_err[index] /= (T * size)
            true_err[index] /= T

        plt.plot(m_arr, empirical_err, label='Empirical Error')
        plt.plot(m_arr, true_err, label='Expected True Error', color="red")
        plt.legend()
        plt.xlabel("Sample size")
        plt.ylabel("Size of Error")
        plt.show()

        return np.stack((empirical_err, true_err))

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals_list in the first experiment.
               m_last - an integer, the maximum number of intervals_list in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_arr = np.arange(k_first, k_last + 1, step)
        empirical_err = np.zeros(len(k_arr))
        true_err = np.zeros(len(k_arr))
        sample = self.sample_from_D(m)
        x_axis, y_axis = sample[:, 0], sample[:, 1]

        for index, k in enumerate(k_arr):
            interval_list, emp_err = intervals.find_best_interval(x_axis, y_axis, k)
            empirical_err[index] += emp_err / m
            true_err[index] += self.calc_true_err(interval_list)

        plt.plot(k_arr, empirical_err, label="Empirical Error")
        plt.plot(k_arr, true_err, label="Expected True Error")
        plt.legend()
        plt.xlabel("Number of Intervals")
        plt.ylabel("Size of Error")
        plt.show()
        return np.argmin(empirical_err)

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals_list in the first experiment.
               m_last - an integer, the maximum number of intervals_list in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_arr = np.arange(k_first, k_last + 1, step)
        empirical_err = np.zeros(len(k_arr))
        true_err = np.zeros(len(k_arr))
        min_penalty, penalty_plus_empirical_error = np.zeros(len(k_arr)), np.zeros(len(k_arr))
        sample = self.sample_from_D(m)
        x_axis, y_axis = sample[:, 0], sample[:, 1]

        for index, k in enumerate(k_arr):
            penalty_func = 2 * (((2 * k + np.emath.log(2 * 10)) / m) ** 0.5)
            interval_list, emp_err = intervals.find_best_interval(x_axis, y_axis, k)
            empirical_err[index] += emp_err / m
            true_err[index] += self.calc_true_err(interval_list)
            min_penalty[index] += penalty_func
            penalty_plus_empirical_error[index] += penalty_func + (emp_err / m)

        plt.plot(k_arr, empirical_err, label="Empirical Error")
        plt.plot(k_arr, true_err, label="Expected True Error")
        plt.plot(k_arr, min_penalty, label="Penalty of the ERM")
        plt.plot(k_arr, penalty_plus_empirical_error, label="Penalty + Empirical Error")
        plt.legend()
        plt.xlabel("Number of Intervals")
        plt.ylabel("Size of Error")
        plt.show()
        return np.argmin(empirical_err)

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        sample = self.sample_from_D(m)
        np.random.shuffle(sample)
        S1, S2 = np.array(sorted(sample[int(m / 5):], key=lambda x: x[0])), sample[:int(m / 5)]
        erm_arr, validation_err_arr = [], np.zeros(10)
        x_axis, y_axis = S1[:, 0], S1[:, 1]

        for k in range(1, 11):
            erm, _ = intervals.find_best_interval(x_axis, y_axis, k)
            erm_arr.append(erm)
            validation_err_arr[k - 1] = self.calc_validation_err(erm, S2)

        return np.argmin(validation_err_arr) + 1

    #################################
    # Place for additional methods

    #################################

    def interval_supplement(self, intervals_list):
        """Finds ([0,1]/intervals_list), the supplement of intervals_list.
        Input: array of array of size 2.

        Returns: array of array of size 2, the rest of the intervals_list
        """
        intervals0 = []
        if intervals_list[0][0] > 0:
            intervals0.append([0.0, intervals_list[0][0]])
        for i in range(len(intervals_list) - 1):
            intervals0.append([intervals_list[i][1], intervals_list[i + 1][0]])
        if intervals_list[-1][1] < 1.0:
            intervals0.append([intervals_list[-1][1], 1.0])
        return intervals0

    def calc_intersect(self, i1, i2):
        """Finds the area of the intersection between two intervals
        Input: i1, i2 are arrays of size 2

        Returns: float in range(0,1)
        """
        # intervals are disjoint
        if i1[1] <= i2[0] or i2[1] <= i1[0]:
            return 0.0

        # i1 contains i2
        if i1[0] <= i2[0] <= i2[1] <= i1[1]:
            return i2[1] - i2[0]

        # i1 contains i2
        if i2[0] <= i1[0] <= i1[1] <= i2[1]:
            return i1[1] - i1[0]

        # Partial intersection
        if i1[0] <= i2[0] and i1[1] <= i2[1]:
            return i1[1] - i2[0]
        if (i1[0] >= i2[0]) and (i1[1] >= i2[1]):
            return i2[1] - i1[0]

    def calc_true_err(self, interval_list):
        """Finds the expected true error given a hypothesis
        Input: interval_list- a list of lists of size 2

        Returns: float in range(0,1)
        """
        realizable_intervals = [[0, 0.2], [0.4, 0.6], [0.8, 1]]
        prob1, prob0 = 0.8, 0.1
        out = 0.0
        realizable_intervals_supplement = self.interval_supplement(realizable_intervals)
        interval_list_supplement = self.interval_supplement(interval_list)

        ### p[Y=0|h(x)=1] - probability of the label being '0' while out hypothesis being '1'
        for interval in interval_list:
            for i1 in realizable_intervals_supplement:
                out += self.calc_intersect(interval, i1) * (1 - prob0)
            for i2 in realizable_intervals:
                out += self.calc_intersect(interval, i2) * (1 - prob1)

        ### p[Y=1|h(x)=0] - probability of the label being '1' while out hypothesis being '0'
        for interval in interval_list_supplement:
            for i1 in realizable_intervals_supplement:
                out += self.calc_intersect(interval, i1) * prob0
            for i2 in realizable_intervals:
                out += self.calc_intersect(interval, i2) * prob1

        return out

    def calc_validation_err(self, interval_list, holdout_set):
        """Finds the validation error given a hypothesis and a holdout set
        Input: interval_list- a list of lists of size 2
               holdout_set- a set of points

        Returns: float in range(0,1)- the validation error
        """
        x_axis, y_axis = holdout_set[:, 0], holdout_set[:, 1]
        error = 0.0
        for i in range(len(x_axis)):
            x, y = x_axis[i], y_axis[i]
            for interval in interval_list:
                prediction = 0
                if interval[0] <= x <= interval[1]:
                    prediction = 1
                    break
            error += int(prediction != y)

        return error / len(holdout_set)


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
