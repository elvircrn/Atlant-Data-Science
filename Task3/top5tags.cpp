#include <bits/stdc++.h>

using namespace std;

int main() {
    fstream fgen("Data/genome-scores.csv");
    ofstream gout("Data/movie-gtags.csv");
    ios_base::sync_with_stdio(false);

    string line;

    vector<pair<double, int>> tags;
    tags.reserve(10);

    // Header
    getline(fgen, line);

    int movieId, tagId;
    double score;
    char c[2];

    gout << "movieId";
    for (int i = 0; i < 5; i++)
        gout << ",gtag" + to_string(i);
    gout << '\n';

    while(getline(fgen, line)) {
        stringstream ss(line);
        ss >> movieId;
        ss.read(c, 1);
        ss >> tagId;
        ss.read(c, 1);
        ss >> score;

        tags.emplace_back(score, tagId);
        sort(tags.begin(), tags.end(),
            [](auto x, auto y) -> bool { return x.first > y.first; });

        if (tags.size() > 5)
            tags.pop_back();

        if (tagId == 1128) {
            gout << movieId;
            for (int j = 0; j < 5; j++)
                gout << "," << tags[j].second;
            gout << '\n';
            tags.clear();
        }
    }


    return 0;
}
