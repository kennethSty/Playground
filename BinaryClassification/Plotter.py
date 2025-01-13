import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, loss_values):
        self.loss_values = [i.detach().cpu() for i in loss_values]
        self.training_steps = [i for i in range(len(loss_values))]

    def plot_convergence(self):
        plt.figure(figsize = (10, 6))

        plt.plot(self.loss_values)

        plt.title('Training Loss Convergence')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')

        plt.grid(True)
        plt.legend(loc='upper right')
        plt.tight_layout()
        
        plt.show()
