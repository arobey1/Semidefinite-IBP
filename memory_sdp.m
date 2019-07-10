function [opt] = memory_sdp(lowers, uppers, weights, biases, c, net_dims)

    n0 = net_dims(1); n1 = net_dims(2); n2 = net_dims(3);

    W0 = weights.w0;
    W1 = weights.w1;
    
    b0 = biases.b0;
    b1 = biases.b1;
    
    x0_lower = lowers.l0;
    x1_lower = lowers.l2;
    
    x0_upper = uppers.u0;
    x1_upper = uppers.u2;

    cvx_begin sdp
    cvx_solver mosek
    
    variable x0(n0,1) 
    variable x1(n1,1)
    variable x2(n2,1)
    variable Q0(n0,n0) symmetric
    variable R0(n1,n0)
    variable Q1(n1,n1) symmetric
    variable R1(n2,n1)
    variable Q2(n2,n2) symmetric
    
    maximize( dot(c, x2) )
    subject to
        diag(Q1) == diag(R0 * W0' + x1 * b0');
        diag(Q2) == diag(R1 * W1' + x2 * b1');
        
        x1 >= max(0, W0 * x0 + b0);
        x2 >= max(0, W1 * x1 + b1);
        
        x0_lower <= x0;
        x0 <= x0_upper;
        
        x1_lower <= x1;
        x1 <= x1_upper;
        
        [   Q0  R0' x0; ...
            R0  Q1  x1; ...
            x0' x1' 1   ] >= 0;
        
        [   Q1  R1' x1; ...
            R1  Q2  x2; ...
            x1' x2' 1   ] >= 0;
        
    cvx_end

    opt = cvx_optval;

end