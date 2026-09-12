
py::array_t<double> py_adaptive_integrate_traction(
    py::array_t<double> obs_points, 
    py::array_t<double> source_tris, 
    py::array_t<double> delta_u_list, 
    double nu, double mu,
    bool parallel) // 
{
    //  ()
    auto obs_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(obs_points);
    auto src_c = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(source_tris);
    auto du_c  = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(delta_u_list);

    auto obs_info = obs_c.request();
    auto src_info = src_c.request();
    auto du_info  = du_c.request();

    size_t n_obs = obs_info.shape[0];
    size_t n_src = src_info.shape[0];
    
    // ： du  1x3 ； src 
    size_t n_du = (du_info.ndim == 1) ? 1 : du_info.shape[0];

    // : N_src x N_obs x 6
    std::vector<ssize_t> shape_res = { (ssize_t)n_src, (ssize_t)n_obs, 6 };
    auto result = py::array_t<double>(shape_res);
    py::buffer_info res_info = result.request();

    //  Python  (GIL)
    {
        py::gil_scoped_release release;
        compute_traction_batch_core(
            (const double*)obs_info.ptr, n_obs,
            (const double*)src_info.ptr, n_src,
            (const double*)du_info.ptr, n_du,
            nu, mu,
            (double*)res_info.ptr,
            parallel // 
        );
    }

    return result;
}

// 
PYBIND11_MODULE(dislocation_lib, m) {
    m.def("compute_dislocation_stresses", &py_adaptive_integrate_traction,
          "Adaptive dislocation stresses caused by dislocations (returns N_src x N_obs x 6)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("delta_u_list"),
          py::arg("nu"),
          py::arg("mu"),
          py::arg("parallel") = true);
    
    m.def("compute_dislocation_displacements", &py_compute_adaptive_displacements,
          "Adaptive Compute displacements caused by dislocations (N x M x 3)",
          py::arg("obs_points"),
          py::arg("source_tris"),
          py::arg("delta_u_list"),
          py::arg("lam"),
          py::arg("mu"));
}

