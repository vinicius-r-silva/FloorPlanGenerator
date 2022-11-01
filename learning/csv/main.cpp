#include <iostream>
#include <string.h>
# include "csv.hpp"

int main(){ 
    csv::CSVReader reader("adj.csv");
    // std::cout << "reader size"

    std::vector<std::string> r_names;
    std::vector<std::string> cols = reader.get_col_names();
    // std::vector<std::string> rows = row.get_col_names();
    for(std::string col : cols){
        if(col.size() > 1)
            r_names.push_back(col);
    }

    for(std::string r_name : r_names){
        std::cout << r_name << std::endl;
    }

    // for (csv::CSVRow& row: reader) { // Input iterator
    //     std::vector<std::string> cols = row.get_col_names();
    //     // std::vector<std::string> rows = row.get_col_names();
    //     for(std::string col : cols){
    //         std::cout << col << std::endl;
    //     }
    //     // std::cout << row.size() << std::endl;
    //     int n_cols = row.size();

        
    //     // for (csv::CSVField& field: row) {
    //     for(int i = 0; i < n_cols; i++){
    //         csv::CSVField field = row[i];
    //         // By default, get<>() produces a std::string.
    //         // A more efficient get<string_view>() is also available, where the resulting
    //         // string_view is valid as long as the parent CSVRow is alive
    //         std::cout << cols[i] << "\t" << field.get<>() << std::endl;
    //     }
    // }
}