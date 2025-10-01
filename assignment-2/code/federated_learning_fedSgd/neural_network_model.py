
import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, seed=42, lambd=0.0):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes          # e.g., [D, h1, h2, ..., C]
        self.num_layers = len(layer_sizes)
        self.lambd = lambd
        self.weights = []
        for i in range(self.num_layers - 1):
            # He init, bias included in last dim (+1)
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i] + 1) * np.sqrt(2.0 / layer_sizes[i])
            self.weights.append(w)

    def sigmoid_function(self, z):
        z = np.array(z, dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative_function(self, a):
        return a * (1.0 - a)

    def softmax_function(self, z):
        z = z - z.max(axis=0, keepdims=True)        # stable softmax
        ez = np.exp(z)
        return ez / np.clip(ez.sum(axis=0, keepdims=True), 1e-12, None)

    def forward_pass(self, X, use_softmax_output=True):
        pre_acts, acts = [], []
        A = X
        # hidden layers (sigmoid)
        for l, W in enumerate(self.weights[:-1]):
            A_bias = np.vstack([np.ones((1, A.shape[1])), A])   # (D+1, N)
            Z = W @ A_bias                                      # (H, N)
            A = self.sigmoid_function(Z)
            pre_acts.append(Z); acts.append(A)
        # output layer
        Wout = self.weights[-1]
        A_bias = np.vstack([np.ones((1, A.shape[1])), A])
        ZL = Wout @ A_bias
        if use_softmax_output:
            AL = self.softmax_function(ZL)                      # (C, N)
        else:
            AL = self.sigmoid_function(ZL)                      # keep option for binary
        pre_acts.append(ZL); acts.append(AL)
        return pre_acts, acts

    def compute_cost_multiclass(self, AL, Y_onehot):

        N = Y_onehot.shape[1]
        loss = -np.sum(Y_onehot * np.log(np.clip(AL, 1e-12, None))) / N
        if self.lambd > 0:
            reg = 0.0
            for W in self.weights:
                reg += np.sum(W[:, 1:] ** 2)                   # exclude bias col
            loss += (self.lambd / (2.0 * N)) * reg
        return float(loss)

    def backward_pass_multiclass(self, acts, X, Y_onehot):

        N = X.shape[1]
        grads = [np.zeros_like(W) for W in self.weights]

        H_list = acts[:-1]  # [A1, A2, ..., A_{L-1}]
        AL = acts[-1]  # (C, N)

        dZ = AL - Y_onehot  # (C, N) for softmax+CE
        A_prev = X if len(H_list) == 0 else H_list[-1]
        Ab_prev = np.vstack([np.ones((1, A_prev.shape[1])), A_prev])
        grads[-1] = (dZ @ Ab_prev.T) / N
        if self.lambd > 0:
            reg = self.weights[-1].copy()
            reg[:, 0] = 0
            grads[-1] += (self.lambd / N) * reg

        dA = self.weights[-1][:, 1:].T @ dZ  # (units_{L-1}, N)

        for l in reversed(range(len(H_list))):
            A_hidden = H_list[l]  # (H_l, N)
            A_prev = X if l == 0 else H_list[l - 1]  # feeding activation
            dZ = dA * self.sigmoid_derivative_function(A_hidden)  # (H_l, N)
            Ab_prev = np.vstack([np.ones((1, A_prev.shape[1])), A_prev])
            grads[l] = (dZ @ Ab_prev.T) / N
            if self.lambd > 0:
                reg = self.weights[l].copy()
                reg[:, 0] = 0
                grads[l] += (self.lambd / N) * reg
            if l > 0:
                dA = self.weights[l][:, 1:].T @ dZ  # propagate to previous hidden

        return grads

    def update_parameters(self, grads, lr):
        for l in range(len(self.weights)):
            self.weights[l] -= lr * grads[l]

    # def train_multiclass(self, X, Y_labels, lr=0.1, max_epochs=200, tol=1e-6, verbose=False):
    #     """
    #     X: (D, N)
    #     Y_labels: (N,) integer class ids
    #     """
    #     C = int(np.max(Y_labels)) + 1
    #     Y_onehot = np.zeros((C, Y_labels.shape[0]), dtype=np.float64)
    #     Y_onehot[Y_labels, np.arange(Y_labels.shape[0])] = 1.0
    #
    #     prev = 1e30
    #     for ep in range(max_epochs):
    #         _, acts = self.forward_pass(X, use_softmax_output=True)
    #         loss = self.compute_cost_multiclass(acts[-1], Y_onehot)
    #         grads = self.backward_pass_multiclass(acts, X, Y_onehot)
    #         self.update_parameters(grads, lr)
    #         if verbose and ep % 50 == 0:
    #             print(f"[epoch {ep}] loss={loss:.6f}")
    #         if abs(prev - loss) < tol:
    #             break
    #         prev = loss

    def train_multiclass(self, X, Y_labels, lr=0.1, max_epochs=200, tol=1e-6, verbose=False):
        N = Y_labels.shape[0]
        C = self.layer_sizes[-1]  # <-- GLOBAL number of classes (e.g., 5), not local max

        valid = (Y_labels >= 0) & (Y_labels < C)
        if not np.all(valid):
            X = X[:, valid]
            Y_labels = Y_labels[valid]
            N = Y_labels.shape[0]

        Y_onehot = np.zeros((C, N), dtype=np.float64)
        Y_onehot[Y_labels, np.arange(N)] = 1.0

        prev = 1e30
        for ep in range(max_epochs):
            _, acts = self.forward_pass(X, use_softmax_output=True)
            loss = self.compute_cost_multiclass(acts[-1], Y_onehot)
            grads = self.backward_pass_multiclass(acts, X, Y_onehot)
            self.update_parameters(grads, lr)
            if verbose and ep % 50 == 0:
                print(f"[epoch {ep}] loss={loss:.6f}")
            if abs(prev - loss) < tol:
                break
            prev = loss

    def predict_multiclass(self, X):
        _, acts = self.forward_pass(X, use_softmax_output=True)
        return np.argmax(acts[-1], axis=0)     # (N,)

    def get_params_vector(self):
        flats = [W.reshape(-1) for W in self.weights]
        return np.concatenate(flats, axis=0)

    def set_params_vector(self, vec):
        offset = 0
        for l in range(len(self.weights)):
            shape = self.weights[l].shape
            size = shape[0] * shape[1]
            self.weights[l] = vec[offset:offset+size].reshape(shape)
            offset += size