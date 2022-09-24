#ifndef COMBINATION
#define COMBINATION

template <typename Iterator>
inline bool next_combination(const Iterator first, Iterator k, const Iterator last){
    /* Credits: Thomas Draper */
    if ((first == last) || (first == k) || (last == k))
        return false;
    Iterator itr1 = first;
    Iterator itr2 = last;
    ++itr1;
    if (last == itr1)
        return false;
    itr1 = last;
    --itr1;
    itr1 = k;
    --itr2;
    while (first != itr1)
    {
        if (*--itr1 < *itr2)
        {
            Iterator j = k;
            while (!(*itr1 < *j)) ++j;
            std::iter_swap(itr1,j);
            ++itr1;
            ++j;
            itr2 = k;
            std::rotate(itr1,j,last);
            while (last != j)
            {
                ++j;
                ++itr2;
            }
            std::rotate(k,itr2,last);
            return true;
        }
    }
    std::rotate(first,k,last);
    return false;
}

std::vector<std::vector<RoomConfig>> getAllComb(std::vector<RoomConfig> setups, std::size_t k){
    std::vector<std::vector<RoomConfig>> result = std::vector<std::vector<RoomConfig>>();

    std::size_t n = setups.size();
    std::vector<int> setupIdx;
    for (int i = 0; i < n; setupIdx.push_back(i++));

    do {
        std::vector<RoomConfig> comb = std::vector<RoomConfig>();
        for (int i = 0; i < k; ++i){
            comb.push_back(setups[setupIdx[i]]);
        }
        result.push_back(comb);
    } while(next_combination(setupIdx.begin(),setupIdx.begin() + k, setupIdx.end()));
    return result;
}

#endif //COMBINATION