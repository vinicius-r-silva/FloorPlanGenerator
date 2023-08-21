#ifndef PROCESS_CUDA
#define PROCESS_CUDA

class CudaProcess{
public:
    static void processResult(std::vector<int>& result, const int *h_res, const int res_size, std::vector<int>& h_begin, std::vector<int>& index_table, const int max_layout_size);
};

#endif //PROCESS_CUDA