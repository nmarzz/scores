function[saved_x,saved_t] = sde_solve(f,g,x0,t0,te,h)    
    t = t0;    
    x = x0;
    sqrth = sqrt(h);    
    saved_x = [];
    saved_t = [];

    while t < te
        saved_x = [saved_x ; x];
        saved_t = [saved_t ; t];
        t = t +  h;
        z = randn(size(x));

        x = x + f(x,t) *h + g(x,t)*z * sqrth;
    end    
    
    
end