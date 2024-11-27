import mnist_loader
import tools

# trains, validates, tests = mnist_loader.load_data()
# tools.show_pic(trains[0][0], trains[1][0])

trains, validates, tests = mnist_loader.load_data_wrapper()
tools.show_pic(trains[0][0], trains[0][2])
