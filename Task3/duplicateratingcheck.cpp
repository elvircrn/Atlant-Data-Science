#include <bits/stdc++.h>


/**
Checks if dataset contains user that have rated the same movie
several times.
Conclusion: There are no duplicates.
*/
int main()
{
    freopen("Data/ratings.csv", "r", stdin);

    std::ios_base::sync_with_stdio(false);

    std::set<std::pair<int, int>> s;

    // Line buffer
    std::string line;

    // Pandas header
    std::getline(std::cin, line);

    while (std::getline(std::cin, line))
    {
        std::stringstream ss(line);
        int userId, movieId;
        char c[2];
        ss >> userId;
		// Comma skip
        ss.read(c, 1);
        ss >> movieId;
        if (s.find({userId, movieId}) != s.end())
        {
            std::cout << "Duplicates found: " << userId << ' ' << movieId << '\n';
            break;
        }
        s.insert({ userId, movieId });
    }

    return 0;
}
