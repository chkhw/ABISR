# Specify fixed domains in which to partially run the model
#
# The easiest way to figure these out for your application is to plot a L1b Band-2
#     image and see which row and column indicies are right for your domain.
#
##### These indicies MUST be integers and divisible by 4. ######

domain_inds = {
    'FD_Full': [0, 21696, 0, 21696],
    'FD_Example1': [10000,11000,10000,11000],
    'FD_Example2': [13000,14000,10000,14000],
    'FD_Example3': [10000,13000,0,20000],
    'FD_Example4': [10000,16000,10000,20000],
    # add your domain here
    # [lower_row, upper_row, left_column, right_column]
}
