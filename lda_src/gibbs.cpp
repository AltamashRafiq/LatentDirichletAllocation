
<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['eigen']
setup_pybind11(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <random>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <tuple>
#include <iostream>

namespace py = pybind11;

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> griffiths_steyvers(int max_iter, float alpha, float beta,
                                                                int D, int W, int T,
                                                                Eigen::VectorXd docs_nwords, Eigen::VectorXd all_words, Eigen::VectorXd ta,
                                                                Eigen::MatrixXd wt, Eigen::MatrixXd dt){
    
  int t0, t1, i, d, w, wid, ta_idx, nwords;
    
  Eigen::VectorXd p_dt, p_wt, unnormalized_p_z, p_z;
    
  std::random_device rd;
  std::mt19937 gen(rd());
    
  for (i = 0; i < max_iter; i++){

      ta_idx = 0;
      
    for (d = 0; d < D; d++) {
        
        nwords = docs_nwords(d);
        
      for (w = 0; w < nwords; w++){
          
        t0 = ta(ta_idx); 
        wid = all_words(ta_idx); 

        dt(d, t0) = dt(d, t0) - 1;

        wt(t0, wid) = wt(t0, wid) - 1;                                
                                                                      
        p_dt = (alpha + dt.row(d).array()) / (dt.row(d).sum() + T * alpha);     
        p_wt = (wt.col(wid).array() + beta) / (wt.rowwise().sum().array() + W * beta); 
                                                                      
        unnormalized_p_z = p_dt.array() * p_wt.array();
          
        p_z = unnormalized_p_z.array() / unnormalized_p_z.array().sum();   
          
        std::discrete_distribution<int> dist(p_z.data(), p_z.data() + p_z.size());
          
        t1 = dist(gen);
          
        ta(ta_idx) = t1;                                                
                                                                      
        dt(d, t1) = dt(d, t1) + 1;                                    
        wt(t1, wid) = wt(t1, wid) + 1;
          
        ta_idx += 1;
      }
    }
    if (i  == 0) {
      py::print("Printing every % 10 iterations");
    }
    if ((i % 10) == 0) {
      py::print("iteration ", i);
    }
  }
  return std::make_tuple(wt, dt);
}

                    
PYBIND11_MODULE(gibbs, m) {
    m.doc() = "auto-compiled c++ extension";
    m.def("griffiths_steyvers", &griffiths_steyvers);
}
