import numpy as np
import matlab.engine

from compute_bounds import compute_bounds

def main():
    eng = matlab.engine.start_matlab()
    lowers, uppers, weights, biases = compute_bounds()

    lowers, uppers = create_bounds_dict(lowers, uppers)
    weights, biases = create_affine_dict(weights, biases)

    id = np.eye(2)
    c = matlab.double(id[:, 0].tolist())
    net_dims = matlab.double([2, 3, 2])

    a = eng.memory_sdp(lowers, uppers, weights, biases, c, net_dims, nargout=1)


def create_bounds_dict(lowers, uppers):

    lower_dict, upper_dict = {}, {}
    for idx, (l, u) in enumerate(zip(lowers, uppers)):
        lower_dict['l' + str(idx)] = matlab.double(l.tolist())
        upper_dict['u' + str(idx)] = matlab.double(u.tolist())

    return lower_dict, upper_dict

def create_affine_dict(weights, biases):

    weight_dict, bias_dict = {}, {}
    for idx, (w, b) in enumerate(zip(weights, biases)):
        weight_dict['w' + str(idx)] = matlab.double(w.tolist())
        bias_dict['b' + str(idx)] = matlab.double(b.tolist())

    return weight_dict, bias_dict


if __name__ == '__main__':
    main()
